import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_campaign

# run_campaign puts the repository root on sys.path when it is imported.
from ops.gpuq import JobSpec, QueueDatabase

RUN_DIR_RELATIVE = "tracking/runs/test-campaign/fgsm-smoke"
RUN_ARTIFACTS = ("resolved_config.yaml", "summary.json", "norm_audit.json")


def _write_status(path: Path, **overrides):
    """Write a schema-shaped attack status, defaulting to a provisional queued."""

    status = {
        "schema_version": 1,
        "task_id": "fgsm-smoke",
        "attack": "fgsm",
        "scope": "smoke",
        "outcome": "queued",
        "decision": "pending",
        "summary": "CPU checks passed; the experiment was queued.",
    }
    status.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status), encoding="utf-8")
    return status


def _prepare_queue(root: Path) -> tuple[QueueDatabase, str, Path]:
    """Build a real SQLite queue on a temporary project root; no GPU involved."""

    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "exp.yaml").write_text(
        "experiment_id: fgsm-smoke\n", encoding="utf-8"
    )
    database = QueueDatabase(root / ".gpuq", root)
    spec = JobSpec.from_mapping(
        {
            "schema_version": 1,
            "task_kind": "attack-experiment",
            "config_path": "configs/exp.yaml",
            "run_dir": RUN_DIR_RELATIVE,
            "requested_memory_mb": 1024,
            "timeout_seconds": 60,
        },
        root,
    )
    record, _ = database.submit(spec)
    return database, record["id"], root / RUN_DIR_RELATIVE


def _drive_to_running(database: QueueDatabase, job_id: str) -> dict:
    database.claim(job_id, "GPU-test-0001", "test-owner")
    return database.mark_running(job_id, 424242, "log.txt")


def _drive_to_succeeded(database: QueueDatabase, job_id: str) -> dict:
    _drive_to_running(database, job_id)
    database.mark_validating(job_id)
    return database.mark_succeeded(job_id)


def _write_run_artifacts(run_dir: Path, verifier_outcome: str | None) -> Path:
    """Create the attempt directory the scheduler would have produced."""

    attempt = run_dir / "attempt-0001"
    attempt.mkdir(parents=True, exist_ok=True)
    for name in RUN_ARTIFACTS:
        (attempt / name).write_text("{}\n", encoding="utf-8")
    if verifier_outcome is not None:
        (attempt / "verification.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "outcome": verifier_outcome,
                    "errors": []
                    if verifier_outcome == "passed"
                    else ["post-save L-inf budget violated"],
                }
            ),
            encoding="utf-8",
        )
    return attempt


class CampaignTests(unittest.TestCase):
    def test_manifest_is_valid_and_ordered(self):
        manifest = run_campaign.load_manifest(run_campaign.DEFAULT_MANIFEST)
        tasks = manifest["profiles"]["development"]["tasks"]
        self.assertEqual(tasks[0]["id"], "ifgsm-smoke")
        self.assertEqual(tasks[-1]["id"], "select-finalists")

    def test_full_tasks_are_unique_and_sequential(self):
        tasks = run_campaign.build_full_tasks(["mifgsm", "dd-fcma"], 2)
        self.assertEqual(
            [task["id"] for task in tasks], ["mifgsm-full", "dd-fcma-full"]
        )
        self.assertEqual(tasks[1]["after"], ["mifgsm-full"])

    def test_full_tasks_reject_duplicates(self):
        with self.assertRaises(run_campaign.CampaignError):
            run_campaign.build_full_tasks(["mifgsm", "mifgsm"], 2)

    def test_full_tasks_reject_unknown_attacks(self):
        with self.assertRaises(run_campaign.CampaignError):
            run_campaign.build_full_tasks(["unknown"], 2, {"mifgsm"})

    def test_state_contains_frozen_manifest_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "CAMPAIGN.yaml"
            manifest.write_bytes(run_campaign.DEFAULT_MANIFEST.read_bytes())
            runs_root = root / "runs"
            latest_file = root / "latest.json"
            with (
                mock.patch.object(run_campaign, "RUNS_ROOT", runs_root),
                mock.patch.object(run_campaign, "LATEST_FILE", latest_file),
            ):
                state, state_path = run_campaign.create_state(
                    "development", [], manifest
                )

            snapshot = Path(state["manifest"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(snapshot.parent.resolve(), state_path.parent.resolve())
            self.assertEqual(
                state["manifest_sha256"], run_campaign.sha256_file(snapshot)
            )

    def test_worker_may_not_claim_passed(self):
        task = {"id": "fgsm-smoke", "attack": "fgsm", "scope": "smoke"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            result = root / "result.json"
            config.write_text("attack: fgsm\n", encoding="utf-8")
            result.write_text("{}\n", encoding="utf-8")
            status_path = root / "status.json"
            _write_status(
                status_path,
                outcome="passed",
                configs=[str(config)],
                results=[str(result)],
                evidence=[str(result)],
                verifier_report=str(result),
            )
            with self.assertRaises(run_campaign.CampaignError) as caught:
                run_campaign.validate_attack_status(status_path, task)
            self.assertIn("controller-owned outcome", str(caught.exception))

    def test_queued_status_requires_a_job_id(self):
        task = {"id": "fgsm-smoke", "attack": "fgsm", "scope": "smoke"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            _write_status(status_path, outcome="queued", job_spec="tracking/jobs/x.json")
            with self.assertRaises(run_campaign.CampaignError):
                run_campaign.validate_attack_status(status_path, task)

    def test_status_task_id_must_match_the_campaign_task(self):
        task = {"id": "fgsm-smoke", "attack": "fgsm", "scope": "smoke"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            _write_status(status_path, task_id="fgsm-development", outcome="blocked")
            with self.assertRaises(run_campaign.CampaignError) as caught:
                run_campaign.validate_attack_status(status_path, task)
            self.assertIn("task_id", str(caught.exception))

    def test_retry_resets_selected_and_later_tasks(self):
        tasks = run_campaign.build_full_tasks(["mifgsm", "dd-fcma"], 2)
        for task in tasks:
            task["state"] = "failed"
            task["status_file"] = "old.json"
        state = {"tasks": tasks, "status": "failed"}
        run_campaign.reset_for_retry(state, "dd-fcma-full")
        self.assertEqual(tasks[0]["state"], "failed")
        self.assertEqual(tasks[1]["state"], "pending")
        self.assertNotIn("status_file", tasks[1])

    def test_full_finalist_requires_passed_development(self):
        state = {
            "tasks": [
                {
                    "kind": "attack",
                    "attack": "mifgsm",
                    "scope": "development",
                    "state": "passed",
                }
            ]
        }
        run_campaign.validate_finalists_development(state, ["mifgsm"])
        with self.assertRaises(run_campaign.CampaignError):
            run_campaign.validate_finalists_development(state, ["dd-fcma"])

if __name__ == "__main__":
    unittest.main()


class QueuedLifecycleTests(unittest.TestCase):
    """The controller owns queued -> running -> verification -> terminal."""

    def _campaign(self, root: Path, job_id: str, **arg_overrides):
        """Drive one required attack task through a full controller run."""

        manifest = run_campaign.load_manifest(run_campaign.DEFAULT_MANIFEST)
        tasks = run_campaign.materialize_tasks(
            [
                {
                    "id": "fgsm-smoke",
                    "attack": "fgsm",
                    "scope": "smoke",
                    "required": True,
                }
            ]
        )
        campaign_dir = root / "campaign-run"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        state_path = campaign_dir / "state.json"
        state = {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "profile": "development",
            "status": "running",
            "created_at": run_campaign.utc_now(),
            "updated_at": run_campaign.utc_now(),
            "manifest": str(run_campaign.DEFAULT_MANIFEST),
            "run_dir": str(campaign_dir),
            "finalists": [],
            "tasks": tasks,
        }
        run_campaign.save_state(state_path, state)

        def fake_child(command, _log_path):
            # argv tail is <attack> <scope> <status-file> <task-id>.
            self.assertEqual(command[-1], "fgsm-smoke")
            _write_status(
                Path(command[-2]),
                outcome="queued",
                job_id=job_id,
                job_spec="tracking/jobs/fgsm-smoke/job-spec.json",
            )
            return 0

        args = SimpleNamespace(
            model=None, variant=None, auto=False, poll_timeout_seconds=5.0
        )
        for key, value in arg_overrides.items():
            setattr(args, key, value)
        with (
            mock.patch.object(run_campaign, "ROOT", root),
            mock.patch.object(run_campaign, "POLL_INTERVAL_SECONDS", 0.01),
            mock.patch.object(run_campaign, "validate_preflight"),
            mock.patch.object(
                run_campaign.shutil, "which", return_value="/usr/bin/opencode"
            ),
            mock.patch.object(run_campaign, "run_process", side_effect=fake_child),
        ):
            exit_code = run_campaign.run_campaign(state, state_path, manifest, args)
        return exit_code, state, state_path

    def test_queued_job_with_passing_verifier_becomes_passed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, run_dir = _prepare_queue(root)
            _drive_to_succeeded(database, job_id)
            _write_run_artifacts(run_dir, "passed")

            exit_code, state, _ = self._campaign(root, job_id)

            self.assertEqual(exit_code, 0)
            self.assertEqual(state["status"], "completed")
            task = state["tasks"][0]
            self.assertEqual(task["state"], "passed")
            self.assertEqual(task["job_id"], job_id)

            final = json.loads(Path(task["status_file"]).read_text(encoding="utf-8"))
            self.assertEqual(final["outcome"], "passed")
            self.assertEqual(
                final["verifier_report"],
                f"{RUN_DIR_RELATIVE}/attempt-0001/verification.json",
            )
            self.assertEqual(
                final["results"], [f"{RUN_DIR_RELATIVE}/attempt-0001/summary.json"]
            )
            # The worker's provisional document is kept beside it for audit.
            worker_copy = Path(task["status_file"]).with_name("status.worker.json")
            self.assertTrue(worker_copy.is_file())
            self.assertEqual(
                json.loads(worker_copy.read_text(encoding="utf-8"))["outcome"], "queued"
            )

    def test_succeeded_job_with_failing_verifier_is_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, run_dir = _prepare_queue(root)
            _drive_to_succeeded(database, job_id)
            _write_run_artifacts(run_dir, "failed")

            exit_code, state, _ = self._campaign(root, job_id)

            self.assertEqual(exit_code, 2)
            task = state["tasks"][0]
            self.assertEqual(task["state"], "failed")
            self.assertIn("Verifier rejected the run", task["summary"])
            final = json.loads(Path(task["status_file"]).read_text(encoding="utf-8"))
            self.assertNotIn("verifier_report", final)

    def test_succeeded_job_without_verifier_report_is_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, run_dir = _prepare_queue(root)
            _drive_to_succeeded(database, job_id)
            _write_run_artifacts(run_dir, None)

            exit_code, state, _ = self._campaign(root, job_id)

            self.assertEqual(exit_code, 2)
            self.assertEqual(state["tasks"][0]["state"], "failed")
            self.assertIn("verification.json is missing", state["tasks"][0]["summary"])

    def test_failed_job_reports_the_queue_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, _ = _prepare_queue(root)
            _drive_to_running(database, job_id)
            database.mark_failed(job_id, "runner exceeded its timeout", exit_code=124)

            exit_code, state, _ = self._campaign(root, job_id)

            self.assertEqual(exit_code, 2)
            self.assertEqual(state["tasks"][0]["state"], "failed")
            self.assertIn("runner exceeded its timeout", state["tasks"][0]["summary"])

    def test_cancelled_job_aborts_a_required_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, _ = _prepare_queue(root)
            database.request_cancel(job_id)
            self.assertEqual(database.get(job_id)["state"], "cancelled")

            exit_code, state, _ = self._campaign(root, job_id)

            self.assertEqual(exit_code, 2)
            self.assertEqual(state["tasks"][0]["state"], "cancelled")
            self.assertEqual(state["status"], "cancelled")

    def test_reconcile_recovers_a_controller_killed_mid_poll(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, run_dir = _prepare_queue(root)
            _drive_to_succeeded(database, job_id)
            _write_run_artifacts(run_dir, "passed")

            campaign_dir = root / "campaign-run"
            attempt_dir = campaign_dir / "01-fgsm-smoke-attempt-1"
            status_path = attempt_dir / "status.json"
            _write_status(status_path, outcome="queued", job_id=job_id,
                          job_spec="tracking/jobs/fgsm-smoke/job-spec.json")
            task = {
                "id": "fgsm-smoke",
                "kind": "attack",
                "attack": "fgsm",
                "scope": "smoke",
                "required": True,
                "needs": [],
                "after": [],
                "state": "running",
                "summary": "",
                "job_id": job_id,
                "attempts": [{"attempt": 1, "status_file": str(status_path)}],
            }
            state = {"tasks": [task]}
            state_path = campaign_dir / "state.json"

            with mock.patch.object(run_campaign, "ROOT", root):
                run_campaign.reconcile_running_tasks(database, state, state_path)
                self.assertEqual(task["state"], "passed")

                # A second pass must be a no-op, not a second rewrite.
                snapshot = json.loads(status_path.read_text(encoding="utf-8"))
                run_campaign.reconcile_running_tasks(database, state, state_path)
                self.assertEqual(task["state"], "passed")
                self.assertEqual(
                    json.loads(status_path.read_text(encoding="utf-8")), snapshot
                )

    def test_interrupt_while_polling_never_cancels_the_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, _ = _prepare_queue(root)
            _drive_to_running(database, job_id)

            with mock.patch.object(
                run_campaign.time, "sleep", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_campaign.poll_until_terminal(database, job_id, None)

            record = database.get(job_id)
            self.assertEqual(record["state"], "running")
            self.assertFalse(record["cancel_requested"])

    def test_poll_timeout_leaves_the_job_with_the_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id, _ = _prepare_queue(root)
            _drive_to_running(database, job_id)

            with mock.patch.object(run_campaign, "POLL_INTERVAL_SECONDS", 0.0):
                record = run_campaign.poll_until_terminal(
                    database, job_id, run_campaign.time.monotonic()
                )

            self.assertIsNone(record)
            self.assertEqual(database.get(job_id)["state"], "running")
            self.assertFalse(database.get(job_id)["cancel_requested"])

    def test_retry_clears_the_previous_queue_binding(self):
        tasks = run_campaign.build_full_tasks(["mifgsm"], 1)
        tasks[0].update({"state": "failed", "job_id": "gpuq-old", "job_spec": "x.json"})
        state = {"tasks": tasks, "status": "failed"}
        run_campaign.reset_for_retry(state, "mifgsm-full")
        self.assertEqual(tasks[0]["state"], "pending")
        self.assertNotIn("job_id", tasks[0])
        self.assertNotIn("job_spec", tasks[0])

    def test_structural_fallback_matches_the_schema_decisions(self):
        """The controller must still enforce the contract without jsonschema."""

        task = {"id": "fgsm-smoke", "attack": "fgsm", "scope": "smoke"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "queued.json"
            _write_status(
                good,
                outcome="queued",
                job_id="gpuq-1",
                job_spec="tracking/jobs/fgsm-smoke/job-spec.json",
            )
            bad = root / "no-job.json"
            _write_status(bad, outcome="queued")
            with mock.patch.object(run_campaign, "HAVE_JSONSCHEMA", False):
                status = run_campaign.validate_attack_status(good, task)
                self.assertEqual(status["outcome"], "queued")
                with self.assertRaises(run_campaign.CampaignError):
                    run_campaign.validate_attack_status(bad, task)


class ReviewStatusTests(unittest.TestCase):
    """The finalist reviewer's structured output is controller-validated."""

    TASK = {"id": "select-finalists", "kind": "review", "required": True}
    ALLOWED = {"ifgsm", "mifgsm", "dd-fcma"}

    def _status(self, root: Path, **overrides) -> Path:
        document = {
            "schema_version": 1,
            "task": "select-finalists",
            "outcome": "passed",
            "summary": "Frozen gates applied to verifier-approved runs.",
        }
        document.update(overrides)
        path = root / "review.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_passed_review_requires_known_finalists_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "validation.json"
            evidence.write_text("{}\n", encoding="utf-8")
            path = self._status(
                root, finalists=["mifgsm"], evidence=[str(evidence)]
            )
            status = run_campaign.validate_review_status(
                path, self.TASK, 2, self.ALLOWED
            )
            self.assertEqual(status["finalists"], ["mifgsm"])

    def test_review_rejects_unknown_or_excess_finalists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "validation.json"
            evidence.write_text("{}\n", encoding="utf-8")
            unknown = self._status(
                root, finalists=["not-an-attack"], evidence=[str(evidence)]
            )
            with self.assertRaises(run_campaign.CampaignError):
                run_campaign.validate_review_status(unknown, self.TASK, 2, self.ALLOWED)

            excess = self._status(
                root,
                finalists=["ifgsm", "mifgsm", "dd-fcma"],
                evidence=[str(evidence)],
            )
            with self.assertRaises(run_campaign.CampaignError):
                run_campaign.validate_review_status(excess, self.TASK, 2, self.ALLOWED)

    def test_review_rejects_missing_evidence_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._status(
                root, finalists=["ifgsm"], evidence=["tracking/runs/absent.json"]
            )
            with self.assertRaises(run_campaign.CampaignError):
                run_campaign.validate_review_status(path, self.TASK, 2, self.ALLOWED)

    def test_review_may_not_report_a_queue_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._status(Path(temporary), outcome="queued")
            with self.assertRaises(run_campaign.CampaignError) as caught:
                run_campaign.validate_review_status(path, self.TASK, 2, self.ALLOWED)
            self.assertIn("Invalid review outcome", str(caught.exception))

    def test_blocked_review_needs_only_a_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._status(
                Path(temporary),
                outcome="blocked",
                summary="No development run has a valid verifier report.",
            )
            status = run_campaign.validate_review_status(
                path, self.TASK, 2, self.ALLOWED
            )
            self.assertEqual(status["outcome"], "blocked")
