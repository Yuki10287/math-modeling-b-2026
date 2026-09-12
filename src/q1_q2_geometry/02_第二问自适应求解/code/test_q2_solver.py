"""Q2 的独立几何不变量、原型回归与本地 JSON/CLI 契约测试。

只在测试层加载旧原型的几何函数；不实例化环境，不调用官方客户端。
运行：python -m unittest discover -s src/q1_q2_geometry -p test_q2_solver.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import shutil
import unittest
import uuid

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q2_solver as q2


def _load_prototype():
    spec = importlib.util.spec_from_file_location(
        "_q2_test_original_geometry", ROOT / "archive/q3_prototype/solver.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def direction(position, angle):
    return {"position": list(position), "status": "direction", "angle": angle}


def contains(vertices, points, tolerance=2e-6):
    """独立成员检查：Qhull 半空间，退化点/线段用投影距离。"""
    P = np.asarray(vertices, dtype=float).reshape((-1, 2))
    X = np.asarray(points, dtype=float).reshape((-1, 2))
    if not len(P):
        return not len(X)
    if len(P) == 1 or np.max(np.linalg.norm(P-P[0], axis=1)) < 1e-10:
        return bool(np.all(np.linalg.norm(X-P[0], axis=1) <= tolerance))
    if np.linalg.matrix_rank(P-P.mean(axis=0), tol=1e-8) < 2:
        differences = P[:, None, :]-P[None, :, :]
        i, j = np.unravel_index(np.argmax(np.sum(differences**2, axis=2)),
                                (len(P), len(P)))
        edge = P[j]-P[i]
        t = np.clip(((X-P[i]) @ edge)/(edge @ edge), 0, 1)
        residual = X-(P[i]+t[:, None]*edge)
        return bool(np.all(np.linalg.norm(residual, axis=1) <= tolerance))
    equations = ConvexHull(P).equations
    return bool(np.all(X @ equations[:, :2].T+equations[:, 2] <= tolerance))


class GeometryAssertions(unittest.TestCase):
    def assertSameRegion(self, first, second, tolerance=2e-6):
        self.assertTrue(contains(first, second, tolerance), "第二个区域越出第一个区域")
        self.assertTrue(contains(second, first, tolerance), "第一个区域越出第二个区域")
        self.assertAlmostEqual(q2.core.diameter(first)[0],
                               q2.core.diameter(second)[0], delta=tolerance)


class FirstQuestionGeometryTests(GeometryAssertions):
    def test_point_and_segment_are_valid_degenerate_regions(self):
        point = q2.core.polygon_from_measurements([([0, 0], 0), ([0, 0], 180)])
        self.assertEqual(point["status"], "bounded")
        self.assertAlmostEqual(q2.core.diameter(point["vertices"])[0], 0, delta=1e-7)
        summary = q2.region_summary(np.asarray(point["vertices"]))
        self.assertAlmostEqual(summary["enclosing_circle"]["radius_m"], 0, delta=1e-7)
        segment = q2.core.polygon_from_measurements([
            ([0, 0], 1), ([0, 0], 359), ([10, 0], 179), ([10, 0], 181)])
        self.assertEqual(segment["status"], "bounded")
        self.assertSameRegion(segment["vertices"], [[0, 0], [10, 0]])
        c, r = q2.enclosing_circle(np.asarray(segment["vertices"]))
        np.testing.assert_allclose(c, [5, 0], atol=1e-7)
        self.assertAlmostEqual(r, 5, delta=1e-7)

    def test_triangle_diameter_does_not_imply_same_diameter_cover(self):
        P = np.array([[0, 0], [38, 0], [19, 19*math.sqrt(3)]], dtype=float)
        self.assertAlmostEqual(q2.core.diameter(P)[0], 38, places=8)
        c, r = q2.enclosing_circle(P)
        self.assertAlmostEqual(r, 38/math.sqrt(3), places=8)
        self.assertGreater(r, 20)
        self.assertTrue(np.all(np.linalg.norm(P-c, axis=1) <= r+1e-8))

    def test_empty_and_unbounded_angles_are_distinguished(self):
        self.assertEqual(q2.core.polygon_from_measurements([([0, 0], 0)])["status"],
                         "unbounded")
        result = q2.core.polygon_from_measurements([([1, 0], 0), ([0, 0], 180)])
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(q2.region_summary(np.empty((0, 2))))

    def test_outer_circle_keeps_exact_circle_boundary(self):
        center, radius = np.array([7.5, -12.]), 5.
        angles = np.linspace(0, 2*math.pi, 513)
        boundary = center+radius*np.column_stack((np.cos(angles), np.sin(angles)))
        P = q2.core.outer_disk(center, radius, q2.SIDES)
        self.assertTrue(contains(P, boundary, 1e-8))
        c, r = q2.enclosing_circle(P)
        np.testing.assert_allclose(c, center, atol=1e-8)
        self.assertAlmostEqual(r, radius/math.cos(math.pi/q2.SIDES), places=8)
        square = center+np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
        clipped = q2.disk_clip(square, center, radius)
        self.assertTrue(contains(clipped, boundary, 1e-8))
        self.assertTrue(contains(square, clipped))


class PrototypeRegressionTests(GeometryAssertions):
    CASES = (([0, 0], 0), ([0, 0], 359.7), ([200, -100], 36),
             ([-700, 200], -1), ([600, -600], 137),
             ([-400, -300], 270), ([1700, 0], 179))

    @classmethod
    def setUpClass(cls):
        cls.original = _load_prototype()

    def test_geometry_selection_prediction_and_posterior_match_original(self):
        for position, angle in self.CASES:
            with self.subTest(position=position, angle=angle):
                s = np.asarray(position, dtype=float)
                plan = q2.plan_second_measurement(direction(position, angle))
                self.assertEqual(plan["status"], "ready")
                P = np.asarray(plan["region"]["vertices"])
                old_P = self.original.initial_belief(s, angle % 360)
                self.assertSameRegion(P, old_P)
                old_q = self.original.choose_measure(old_P, s, "geometry")
                selected = np.asarray(plan["selected_point"])
                # 若未来发生镜像同分分叉，此断言会显式失败，不用宽松距离掩盖。
                np.testing.assert_allclose(selected, old_q, rtol=0, atol=2e-6)
                truth = P.mean(axis=0)
                bearing = math.degrees(math.atan2(*(truth-selected)[::-1])) % 360
                for error in (-.75, 0., .75):
                    observation = direction(selected, (bearing+error) % 360)
                    predicted = q2.wedge(P, selected, observation["angle"])
                    old_predicted = self.original.wedge(old_P, selected, observation["angle"])
                    self.assertSameRegion(predicted, old_predicted)
                    result = q2.update_second_measurement(plan, observation)
                    self.assertEqual(result["status"], "direction")
                    posterior = np.asarray(result["region"]["vertices"])
                    old_posterior = self.original.update_belief(old_P, selected,
                                                               observation["angle"])
                    self.assertSameRegion(posterior, old_posterior)
                    self.assertTrue(contains(P, posterior))
                    self.assertTrue(contains(posterior, [truth]))
                    self.assertLessEqual(result["diameter_ratio"], 1+1e-8)
                self.assertTrue(all(c["receive_certificate"]["certified"]
                                    for c in plan["candidates"]))


class ReceptionAndBoundaryTests(GeometryAssertions):
    def test_clipped_intersection_vertices_prevent_false_reception(self):
        P = np.array([[599, -1000], [900, 0], [599, 1000]], dtype=float)
        q, witness = np.array([0., 0.]), np.array([1200., 0.])
        # 全部原顶点都满足变半径式，但中间的新交点不满足。
        self.assertTrue(np.all(np.linalg.norm(P-q, axis=1)
                               <= np.maximum(1000, np.linalg.norm(P-witness, axis=1))))
        cert = q2.reception_certificate(P, q, witness)
        self.assertFalse(cert["certified"])
        self.assertGreater(cert["farther_max_distance_m"], 1100)
        vertices = np.asarray(cert["farther_region_vertices"])
        self.assertTrue(np.any(np.abs(vertices[:, 0]-600) < 1e-8))

    def test_reception_margin_is_conservative_at_circle_boundary(self):
        for distance, expected in ((999.998, True), (999.999, True),
                                   (1000., False), (1000.001, False)):
            with self.subTest(distance=distance):
                P = np.array([[distance, 0.]])
                cert = q2.reception_certificate(P, [0, 0], [1, 0])
                self.assertEqual(cert["certified"], expected)
        # 比首点更近的整片区域无需被1000米圆覆盖。
        cert = q2.reception_certificate(np.array([[1400., 0.]]), [100, 0], [0, 0])
        self.assertTrue(cert["certified"])
        self.assertIsNone(cert["farther_max_distance_m"])

    def test_angle_wrap_is_invariant(self):
        reference = q2.plan_second_measurement(direction([0, 0], 359.7))
        for angle in (-.3, 719.7):
            other = q2.plan_second_measurement(direction([0, 0], angle))
            self.assertSameRegion(reference["region"]["vertices"], other["region"]["vertices"])
            np.testing.assert_allclose(other["selected_point"], reference["selected_point"],
                                       atol=2e-6, rtol=0)
        P = np.asarray(reference["region"]["vertices"])
        angles = np.deg2rad([-1.3, -.3, .7])
        truth = 1000*np.column_stack((np.cos(angles), np.sin(angles)))
        self.assertTrue(contains(P, truth))

    def test_near_first_has_no_second_measurement(self):
        s = np.array([1798., 0.])
        plan = q2.plan_second_measurement({"position": s.tolist(), "status": "near"})
        self.assertEqual(plan["status"], "near")
        self.assertIsNone(plan["selected_point"])
        self.assertEqual(plan["candidates"], [])
        self.assertEqual(plan["near_constraint"]["radius_m"], 5.)
        P = np.asarray(plan["region"]["vertices"])
        self.assertTrue(contains(P, [[1798, 0], [1800, 0], [1793, 0]]))
        self.assertLessEqual(np.max(np.linalg.norm(P-s, axis=1)),
                             5/math.cos(math.pi/q2.SIDES)+1e-7)

    def test_near_second_keeps_truth_and_is_nested(self):
        plan = q2.plan_second_measurement(direction([0, 0], 0), policy="nearest")
        q = plan["selected_point"]
        result = q2.update_second_measurement(plan, {"position": q, "status": "near"})
        self.assertEqual(result["status"], "near")
        P = np.asarray(result["region"]["vertices"])
        self.assertTrue(contains(plan["region"]["vertices"], P))
        self.assertTrue(contains(P, [q]))
        self.assertEqual(result["near_constraint"], {"center": q, "radius_m": 5.})
        self.assertLessEqual(np.max(np.linalg.norm(P-np.asarray(q), axis=1)),
                             5/math.cos(math.pi/q2.SIDES)+1e-7)

    def test_inconsistent_first_and_second_do_not_claim_a_region(self):
        empty = q2.plan_second_measurement(direction([4000, 0], 0))
        self.assertEqual(empty["status"], "inconsistent")
        self.assertIsNone(empty["region"])
        self.assertIsNone(empty["selected_point"])
        plan = q2.plan_second_measurement(direction([0, 0], 0))
        q = np.asarray(plan["selected_point"])
        center = np.mean(plan["region"]["vertices"], axis=0)
        away = math.degrees(math.atan2(*(q-center)[::-1]))
        for observation in (direction(q, away), {"position": q.tolist(), "status": "no_signal"}):
            result = q2.update_second_measurement(plan, observation)
            self.assertEqual(result["status"], "inconsistent")
            self.assertIsNone(result["region"])
            self.assertIsNone(result["diameter_ratio"])


class InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = q2.plan_second_measurement(direction([0, 0], 0))

    def test_malformed_observations_are_rejected(self):
        bad = [None, [], {}, {"position": [0, 0], "status": "direction"},
               direction([0], 0), direction([0, 0, 0], 0), direction([True, 0], 0),
               direction(["0", 0], 0), direction([None, 0], 0),
               direction([float("nan"), 0], 0), direction([1e20, 0], 0),
               direction([[], []], 0), direction([0, 0], "0"),
               direction([0, 0], True), direction([0, 0], float("inf")),
               {"position": [0, 0], "status": "unknown"}]
        for observation in bad:
            with self.subTest(observation=observation):
                with self.assertRaises(ValueError):
                    q2.normalize_observation(observation)
        with self.assertRaises(ValueError):
            q2.plan_second_measurement({"position": [0, 0], "status": "no_signal"})
        with self.assertRaises(ValueError):
            q2.plan_second_measurement(direction([0, 0], 0), policy="time")

    def test_wrong_second_position_and_invalid_plan_are_rejected(self):
        q = np.asarray(self.plan["selected_point"])
        with self.assertRaises(ValueError):
            q2.update_second_measurement(self.plan, direction(q+[.001, 0], 90))
        for changed in ({"schema": "other"}, {"status": "near"},
                        {"angle_error_deg": 2}, {"selected_point": [9000, 9000]}):
            plan = copy.deepcopy(self.plan)
            plan.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                q2.update_second_measurement(plan, direction(plan["selected_point"], 90))

    def test_saved_vertices_are_not_trusted_and_input_plan_is_not_mutated(self):
        plan = copy.deepcopy(self.plan)
        plan["region"]["vertices"] = [[999999, 999999]]
        original = copy.deepcopy(plan)
        q = plan["selected_point"]
        truth = np.array([1000., 0.])
        angle = math.degrees(math.atan2(*(truth-np.asarray(q))[::-1]))
        result = q2.update_second_measurement(plan, direction(q, angle))
        self.assertEqual(result["status"], "direction")
        self.assertTrue(contains(result["region"]["vertices"], [truth]))
        self.assertEqual(plan, original)


class LocalCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT / "tmp/q2_contract_tests").resolve()
        self.folder = self.temp_root / uuid.uuid4().hex
        self.folder.mkdir(parents=True, exist_ok=False)
        self.addCleanup(self.cleanupFolder)

    def cleanupFolder(self):
        target = self.folder.resolve()
        if target.parent != self.temp_root:
            raise RuntimeError("测试清理路径越出预期临时目录")
        shutil.rmtree(target)

    def runCli(self, *arguments):
        return subprocess.run([sys.executable, "-X", "utf8", str(HERE/"q2_solver.py"),
                               *map(str, arguments)], capture_output=True, text=True,
                              encoding="utf-8", timeout=30, cwd=ROOT)

    def writeJson(self, filename, value):
        path = self.folder/filename
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_plan_update_round_trip_and_refusal_to_overwrite(self):
        first = self.writeJson("首测.json", direction([0, 0], 0))
        plan_path, result_path = self.folder/"计划.json", self.folder/"结果.json"
        completed = self.runCli("plan", "--input", first, "--output", plan_path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        saved_bytes = plan_path.read_bytes()
        plan = json.loads(saved_bytes)
        q = np.asarray(plan["selected_point"])
        truth = np.array([1000., 0.])
        angle = math.degrees(math.atan2(*(truth-q)[::-1]))
        second = self.writeJson("二测.json", direction(q, angle))
        completed = self.runCli("update", "--plan", plan_path, "--input", second,
                                "--output", result_path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertTrue(contains(result["region"]["vertices"], [truth]))
        completed = self.runCli("plan", "--input", first, "--output", plan_path)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(plan_path.read_bytes(), saved_bytes)

    def test_invalid_json_and_wrong_position_leave_no_output(self):
        source = self.folder/"bad.json"
        source.write_text('{"position": [0, 0],', encoding="utf-8")
        output = self.folder/"invalid-result.json"
        completed = self.runCli("plan", "--input", source, "--output", output)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(output.exists())
        plan = q2.plan_second_measurement(direction([0, 0], 0))
        plan_path = self.writeJson("plan.json", plan)
        second = self.writeJson("wrong.json", direction([0, 0], 0))
        completed = self.runCli("update", "--plan", plan_path, "--input", second,
                                "--output", output)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(output.exists())

    def test_inconsistent_feedback_produces_only_diagnostic_output(self):
        # 输入合法但观测矛盾应输出结构化诊断；与输入错误退出2且不产文件区分。
        plan = q2.plan_second_measurement(direction([0, 0], 0))
        plan_path = self.writeJson("plan.json", plan)
        second = self.writeJson("negative.json", {
            "position": plan["selected_point"], "status": "no_signal"})
        output = self.folder/"diagnostic.json"
        completed = self.runCli("update", "--plan", plan_path, "--input", second,
                                "--output", output)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        diagnostic = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["status"], "inconsistent")
        self.assertIsNone(diagnostic["region"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
