package com.tzb.safeguard.ui

import com.tzb.safeguard.ui.screens.care.cognitiveAttentionLevelLabel
import com.tzb.safeguard.ui.screens.care.cognitiveStateLabel
import com.tzb.safeguard.ui.screens.care.dataQualityLabel
import com.tzb.safeguard.ui.screens.care.psychologyRiskLevelLabel
import com.tzb.safeguard.ui.screens.fall.decisionPathLabel
import com.tzb.safeguard.ui.screens.fall.riskLevelLabel
import com.tzb.safeguard.ui.screens.fall.sensorLabel
import com.tzb.safeguard.ui.screens.fraud.disposalLabel
import com.tzb.safeguard.ui.screens.fraud.monitoringStatusLabel
import com.tzb.safeguard.ui.screens.home.disposalLabel as homeDisposalLabel
import com.tzb.safeguard.ui.screens.home.fallOverallLabel
import com.tzb.safeguard.ui.screens.home.fallRiskLevelLabel
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * 风险词表一致性测试：锁定 App 各页面映射函数的文案口径，
 * 防止"高风险/需关注/正常/暂无正式判断"等关键词在不同页面漂移。
 */
class LabelMappingTest {

    // ---- 跌倒页 ----

    @Test
    fun fall_risk_level_labels() {
        assertEquals("高风险", riskLevelLabel("critical"))
        assertEquals("高风险", riskLevelLabel("high"))
        assertEquals("需关注", riskLevelLabel("medium"))
        assertEquals("低风险", riskLevelLabel("low"))
        assertEquals("正常", riskLevelLabel("normal"))
        assertEquals("无有效判断", riskLevelLabel("unknown"))
        assertEquals("无有效判断", riskLevelLabel("anything_else"))
    }

    @Test
    fun fall_sensor_labels() {
        assertEquals("已连接 · 监测中 · 数据正常", sensorLabel("available"))
        assertEquals("已连接 · 监测中 · 数据质量一般", sensorLabel("degraded"))
        assertEquals("不适用", sensorLabel("not_applicable"))
        assertEquals("暂不可用 · 暂无数据", sensorLabel("unavailable"))
        assertEquals("状态未知", sensorLabel("weird"))
    }

    @Test
    fun fall_decision_path_labels() {
        assertEquals("视觉主判断 · 雷达证据增强", decisionPathLabel("camera_led_radar_evidence"))
        assertEquals("视觉监测", decisionPathLabel("camera_only"))
        assertEquals("雷达单模态监测", decisionPathLabel("radar_only"))
        assertEquals("监测暂不可用", decisionPathLabel("unavailable"))
    }

    // ---- 首页 ----

    @Test
    fun home_fall_overall_labels() {
        assertEquals("存在需要关注的跌倒风险", fallOverallLabel("critical"))
        assertEquals("存在需要关注的跌倒风险", fallOverallLabel("high"))
        assertEquals("发现风险变化，正在持续监测", fallOverallLabel("medium"))
        assertEquals("各房间正在稳定监测", fallOverallLabel("normal"))
        assertEquals("各房间正在稳定监测", fallOverallLabel("low"))
        assertEquals("服务暂不可用", fallOverallLabel("unknown"))
    }

    @Test
    fun home_fall_risk_level_labels_match_fall_page() {
        // 首页与跌倒页共用同一套风险词表（口径一致性锚点）
        assertEquals(riskLevelLabel("high"), fallRiskLevelLabel("high"))
        assertEquals(riskLevelLabel("medium"), fallRiskLevelLabel("medium"))
        assertEquals(riskLevelLabel("normal"), fallRiskLevelLabel("normal"))
        assertEquals(riskLevelLabel("unknown"), fallRiskLevelLabel("weird"))
    }

    @Test
    fun home_disposal_labels() {
        assertEquals("待处置", homeDisposalLabel("open"))
        assertEquals("处置中", homeDisposalLabel("acknowledged"))
        assertEquals("已完成", homeDisposalLabel("resolved"))
        assertEquals("已标记误报", homeDisposalLabel("false_alarm"))
        assertEquals("weird", homeDisposalLabel("weird"))
    }

    // ---- 诈骗状态页 ----

    @Test
    fun fraud_monitoring_status_labels() {
        assertEquals("运行中", monitoringStatusLabel("running"))
        assertEquals("启动中", monitoringStatusLabel("starting"))
        assertEquals("未启用", monitoringStatusLabel("unavailable"))
        assertEquals("未开启守护", monitoringStatusLabel("stopped"))
        assertEquals("不可用", monitoringStatusLabel("weird"))
    }

    @Test
    fun fraud_disposal_labels_match_home() {
        // 诈骗状态页与首页摘要卡的处置词表一致
        for (s in listOf("open", "acknowledged", "resolved", "false_alarm")) {
            assertEquals(homeDisposalLabel(s), disposalLabel(s))
        }
    }

    // ---- Care 页 ----

    @Test
    fun cognitive_state_labels() {
        assertEquals("正在采集语音资料", cognitiveStateLabel("processing"))
        assertEquals("辅助评估已完成", cognitiveStateLabel("completed"))
        assertEquals("本次分析未完成", cognitiveStateLabel("failed"))
        assertEquals("有效语音资料不足", cognitiveStateLabel("insufficient_data"))
        assertEquals("服务暂不可用", cognitiveStateLabel("unavailable"))
        assertEquals("服务暂不可用", cognitiveStateLabel("weird"))
    }

    @Test
    fun cognitive_attention_level_labels() {
        assertEquals("暂无明显关注", cognitiveAttentionLevelLabel("none"))
        assertEquals("建议关注", cognitiveAttentionLevelLabel("mild"))
        assertEquals("需重点关注", cognitiveAttentionLevelLabel("moderate"))
        assertEquals("高度关注", cognitiveAttentionLevelLabel("high"))
        assertEquals("暂无法判断", cognitiveAttentionLevelLabel("weird"))
    }

    @Test
    fun psychology_risk_level_labels() {
        assertEquals("暂时无风险", psychologyRiskLevelLabel("no_risk"))
        assertEquals("轻度风险", psychologyRiskLevelLabel("mild"))
        assertEquals("中度风险", psychologyRiskLevelLabel("moderate"))
        assertEquals("重度风险", psychologyRiskLevelLabel("severe"))
        assertEquals("暂无法判断", psychologyRiskLevelLabel("weird"))
    }

    @Test
    fun data_quality_labels() {
        assertEquals("良好", dataQualityLabel("usable"))
        assertEquals("有限", dataQualityLabel("limited"))
        assertEquals("不足", dataQualityLabel("insufficient"))
        assertEquals("未知", dataQualityLabel("weird"))
    }
}
