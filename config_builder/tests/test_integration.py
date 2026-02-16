#!/usr/bin/env python
"""
Comprehensive test suite for Config Builder
Tests core functionality: feature params, config building, and file operations
"""

import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8032"


class TestConfigBuilder:
    """Test suite for configuration builder"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0

    def test_health(self):
        """Test health endpoint"""
        print("\n" + "=" * 70)
        print("TEST: Health Check")
        print("=" * 70)

        try:
            r = requests.get(f"{BASE_URL}/health")
            assert r.status_code == 200
            print("PASS: Server is healthy")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_get_features(self):
        """Test GET /api/features endpoint"""
        print("\n" + "=" * 70)
        print("TEST: Get Available Features")
        print("=" * 70)

        try:
            r = requests.get(f"{BASE_URL}/api/features")
            assert r.status_code == 200

            data = r.json()
            features = data.get("features", {})

            # Verify essential features exist
            assert "line_cross" in features
            assert "region_crowd" in features
            assert "dwell_time" in features

            # Verify line_cross includes core + tracking params
            line_cross_params = features["line_cross"]["parameters"]
            assert len(line_cross_params) >= 3
            assert "centroid" in line_cross_params
            assert "lost_threshold" in line_cross_params
            assert "allow_recounting" in line_cross_params
            assert "tracking.tracker_impl" in line_cross_params
            assert "tracking.reid_enabled" in line_cross_params

            print(f"PASS: Found {len(features)} features")
            print(f"  - line_cross: {len(line_cross_params)} parameters")
            print(f"  - region_crowd: {len(features['region_crowd']['parameters'])} parameters")
            print(f"  - dwell_time: {len(features['dwell_time']['parameters'])} parameters")

            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_get_feature_params(self):
        """Test GET /api/feature-params/{feature}"""
        print("\n" + "=" * 70)
        print("TEST: Get Current Feature Parameters")
        print("=" * 70)

        try:
            r = requests.get(f"{BASE_URL}/api/feature-params/line_cross")
            assert r.status_code == 200

            data = r.json()
            params = data.get("params", {})

            # Verify defaults
            assert params.get("centroid") == "mid_centre"
            assert params.get("lost_threshold") == 10
            assert not params.get("allow_recounting")

            print("PASS: Current parameters retrieved")
            for key, value in sorted(params.items()):
                print(f"  - {key}: {value}")

            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_update_feature_params(self):
        """Test POST /api/feature-params/{feature}"""
        print("\n" + "=" * 70)
        print("TEST: Update Feature Parameters")
        print("=" * 70)

        try:
            new_params = {"centroid": "head", "lost_threshold": 20, "allow_recounting": False}

            r = requests.post(f"{BASE_URL}/api/feature-params/line_cross", json=new_params)
            assert r.status_code == 200

            data = r.json()
            result = data.get("params", {})

            # Verify updates
            assert result.get("centroid") == "head"
            assert result.get("lost_threshold") == 20
            assert not result.get("allow_recounting")

            print("PASS: Parameters updated successfully")
            for key in new_params:
                print(f"  - {key}: {result.get(key)}")

            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_reset_feature_params(self):
        """Test DELETE /api/feature-params/{feature}"""
        print("\n" + "=" * 70)
        print("TEST: Reset Parameters to Defaults")
        print("=" * 70)

        try:
            r = requests.delete(f"{BASE_URL}/api/feature-params/line_cross")
            assert r.status_code == 200

            data = r.json()
            result = data.get("params", {})

            # Verify reset
            assert result.get("centroid") == "mid_centre"
            assert result.get("lost_threshold") == 10
            assert not result.get("allow_recounting")

            print("PASS: Parameters reset to defaults")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_build_config(self):
        """Test POST /api/build endpoint"""
        print("\n" + "=" * 70)
        print("TEST: Build Configuration with Feature Parameters")
        print("=" * 70)

        try:
            payload = {
                "config_name": "test_config",
                "model": {
                    "name": "person_360camera_detection_v33",
                    "device": "cpu",
                    "conf": 0.5,
                    "iou": 0.7,
                    "type": "small",
                    "classes": ["person"],
                },
                "feature": "line_cross",
                "feature_params": {
                    "centroid": "head",
                    "lost_threshold": 15,
                    "allow_recounting": True,
                },
                "lines": [
                    {
                        "coords": [{"x": 0.2, "y": 0.3}, {"x": 0.8, "y": 0.7}],
                        "id": 1,
                        "type": "line_1",
                        "color": "#0080ff",
                        "direction": "rightward",
                        "orientation": "diagonal",
                        "bidirectional": False,
                        "mode": [],
                    }
                ],
                "regions": [],
                "source_mode": "inference",
                "cctv_id": "test_cam",
                "max_fps": 30,
            }

            r = requests.post(f"{BASE_URL}/api/build", json=payload)
            assert r.status_code == 200

            yaml_content = r.text

            # Verify YAML contains our parameters
            assert "feature_params:" in yaml_content
            assert "centroid: head" in yaml_content
            assert "lost_threshold: 15" in yaml_content
            assert "allow_recounting: true" in yaml_content

            print("PASS: Configuration built successfully")
            print(f"  YAML length: {len(yaml_content)} bytes")
            print("  Contains parameters: centroid, lost_threshold, allow_recounting")

            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def test_save_config(self):
        """Test POST /api/save endpoint"""
        print("\n" + "=" * 70)
        print("TEST: Save Configuration to File")
        print("=" * 70)

        try:
            payload = {
                "config_name": f"test_save_{int(time.time())}",
                "model": {
                    "name": "person_360camera_detection_v33",
                    "device": "cpu",
                    "conf": 0.5,
                    "iou": 0.7,
                    "type": "small",
                    "classes": ["person"],
                },
                "feature": "line_cross",
                "feature_params": {
                    "centroid": "mid_centre",
                    "lost_threshold": 10,
                    "allow_recounting": True,
                },
                "lines": [
                    {
                        "coords": [{"x": 0.2, "y": 0.3}, {"x": 0.8, "y": 0.7}],
                        "id": 1,
                        "type": "line_1",
                        "color": "#0080ff",
                        "direction": "rightward",
                        "orientation": "diagonal",
                        "bidirectional": False,
                        "mode": [],
                    }
                ],
                "regions": [],
                "source_mode": "inference",
                "cctv_id": "test",
                "max_fps": 30,
            }

            r = requests.post(f"{BASE_URL}/api/save", json=payload)
            assert r.status_code == 200

            data = r.json()
            assert data["success"]
            assert "filename" in data
            assert data["filename"].endswith(".yaml")

            # Verify file exists in configs/app
            config_file = Path("configs/app") / data["filename"]
            time.sleep(0.5)
            assert config_file.exists()

            print("PASS: Configuration saved successfully")
            print(f"  File: {data['filename']}")
            print(f"  Path: {data['path']}")

            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"FAIL: {e}")
            self.tests_failed += 1
            return False

    def run_all(self):
        """Run all tests"""
        print("\n" + "=" * 70)
        print("CONFIG BUILDER TEST SUITE")
        print("=" * 70)

        try:
            self.test_health()
            self.test_get_features()
            self.test_get_feature_params()
            self.test_update_feature_params()
            self.test_reset_feature_params()
            self.test_build_config()
            self.test_save_config()

            # Summary
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            print(f"Passed: {self.tests_passed}")
            print(f"Failed: {self.tests_failed}")

            if self.tests_failed == 0:
                print("\nALL TESTS PASSED!")
                return True
            else:
                print(f"\n{self.tests_failed} TEST(S) FAILED")
                return False

        except Exception as e:
            print(f"\nTEST SUITE ERROR: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    suite = TestConfigBuilder()
    success = suite.run_all()
    sys.exit(0 if success else 1)
