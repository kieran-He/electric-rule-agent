from app.core.province import ProvinceDetector


def test_detect_single_province():
    detector = ProvinceDetector()
    result = detector.detect("2026年陕西电力市场交易流程是什么？")
    assert result.province_code == "SN"
    assert result.confidence > 0.9


def test_detect_no_province():
    detector = ProvinceDetector()
    result = detector.detect("交易流程是什么？")
    assert result.province_code is None
    assert result.confidence == 0.0

