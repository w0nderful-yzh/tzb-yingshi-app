package com.tzb.safeguard.data.network

import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody

/**
 * Mock 拦截器：后端规划接口未落地前，按 docs/api/app-client-api.md 的契约
 * 在本地返回演示数据，保证 App 全流程可跑通。
 * 由 BuildConfig.MOCK_MODE 控制是否启用；联调真实后端时关闭即可。
 */
class MockInterceptor : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val path = request.url.encodedPath
        val method = request.method

        // 模拟网络延迟，便于观察加载态
        Thread.sleep(350)

        val body = when {
            // ---------- 通用 ----------
            path == "/api/v1/users/me" -> ok(USER_JSON)
            path == "/api/v1/safety/status" -> ok(SAFETY_JSON)
            path == "/api/v1/sos" && method == "POST" ->
                ok("""{"event_id":"evt_sos_001","status":"dispatched","notified_contacts":2}""")

            // ---------- 事件 ----------
            path == "/api/v1/events" && method == "GET" -> ok(EVENTS_JSON)
            path.matches(Regex("/api/v1/events/[^/]+")) && method == "GET" -> ok(DETAIL_JSON)
            path.matches(Regex("/api/v1/events/[^/]+/confirm")) && method == "POST" -> ok("{}")
            path.matches(Regex("/api/v1/events/[^/]+/call")) && method == "POST" -> ok("{}")
            path.matches(Regex("/api/v1/events/[^/]+/status")) && method == "PATCH" -> ok("{}")
            // PATCH /events/{id}（无 /status 后缀时也兼容）
            path.matches(Regex("/api/v1/events/[^/]+")) && method == "PATCH" -> ok("{}")

            // ---------- 设备 ----------
            path == "/api/v1/devices" && method == "GET" -> ok(DEVICES_JSON)
            path.matches(Regex("/api/v1/devices/[^/]+/live-url")) ->
                ok("""{"url":"https://example.invalid/live.flv","protocol":"flv","expires_in":300}""")

            // ---------- 家属端 ----------
            path == "/api/v1/family/elders" -> ok(ELDERS_JSON)
            path == "/api/v1/contacts" -> ok(CONTACTS_JSON)
            path == "/api/v1/stats/events" -> ok(STATS_EVENTS_JSON)
            path == "/api/v1/stats/activity" -> ok(STATS_ACTIVITY_JSON)

            else -> error(404, 10002, "Mock 未覆盖的接口: $method $path")
        }

        return Response.Builder()
            .request(request)
            .protocol(Protocol.HTTP_1_1)
            .code(body.first)
            .message(if (body.first == 200) "OK" else "Error")
            .body(body.second.toResponseBody("application/json".toMediaType()))
            .build()
    }

    private fun ok(dataJson: String): Pair<Int, String> =
        200 to """{"code":0,"message":"success","data":$dataJson,"request_id":"req_mock"}"""

    private fun error(http: Int, code: Int, msg: String): Pair<Int, String> =
        http to """{"code":$code,"message":"$msg","data":null,"request_id":"req_mock"}"""

    companion object {
        private const val USER_JSON = """
            {"user_id":"u-elder-001","role":"elder","name":"王秀兰",
             "bound_family_count":2,"font_size":"extra_large","voice_assist_enabled":true}"""

        private const val SAFETY_JSON = """
            {"overall":"danger","overall_label":"有 1 条紧急告警待确认",
             "active_event_count":2,"highest_active_level":"emergency",
             "devices_online":3,"devices_total":3,
             "checked_at":"2026-08-04T16:45:02+08:00",
             "today":{"event_count":2,"active_hours":6.2,"call_screened":1}}"""

        private const val DEVICES_JSON = """
            {"devices":[
              {"device_id":"camera-01","name":"客厅摄像头","room":"living_room","online":true,"signal":"good","last_seen_at":"2026-08-04T16:44:58+08:00"},
              {"device_id":"camera-02","name":"卧室摄像头","room":"bedroom","online":true,"signal":"good","last_seen_at":"2026-08-04T16:44:58+08:00"},
              {"device_id":"camera-03","name":"厨房摄像头","room":"kitchen","online":true,"signal":"weak","last_seen_at":"2026-08-04T16:40:12+08:00"},
              {"device_id":"camera-04","name":"门口摄像头","room":"door","online":false,"signal":"offline","last_seen_at":"2026-08-04T08:10:00+08:00"}
            ]}"""

        private const val EVENTS_JSON = """
            {"events":[
              {"event_id":"evt_001","type":"fall_suspected","level":"emergency","title":"疑似跌倒",
               "summary":"客厅检测到倒地并持续 25 秒未起身","device_id":"camera-01",
               "occurred_at":"2026-08-04T15:02:11+08:00","status":"open","evidence_image_url":null},
              {"event_id":"evt_002","type":"fraud_suspected","level":"warning","title":"疑似诈骗电话",
               "summary":"来电自称“银行客服”并索要短信验证码，风险等级 S4","device_id":"camera-01",
               "occurred_at":"2026-08-04T11:20:05+08:00","status":"open","evidence_image_url":null},
              {"event_id":"evt_003","type":"sedentary","level":"reminder","title":"久坐 1 小时",
               "summary":"建议起身活动，倒杯水","device_id":"camera-01",
               "occurred_at":"2026-08-04T10:32:00+08:00","status":"acknowledged","evidence_image_url":null},
              {"event_id":"evt_004","type":"night_leave_bed","level":"reminder","title":"夜间离床 3 次",
               "summary":"昨夜离床次数略多于平常，注意休息","device_id":"camera-02",
               "occurred_at":"2026-08-04T07:00:00+08:00","status":"acknowledged","evidence_image_url":null},
              {"event_id":"evt_005","type":"stranger","level":"warning","title":"陌生人到访",
               "summary":"门口检测到陌生面孔，停留 4 分钟；儿子已确认：快递员","device_id":"camera-04",
               "occurred_at":"2026-08-03T16:48:00+08:00","status":"resolved","evidence_image_url":null}
            ],"next_cursor":null}"""

        private const val DETAIL_JSON = """
            {"event_id":"evt_001","type":"fall_suspected","level":"emergency","status":"open",
             "device_id":"camera-01","occurred_at":"2026-08-04T15:02:11+08:00",
             "evidence_image_url":null,
             "analysis":{"confidence":0.87,
               "reasons":[
                 {"key":"down_duration_seconds","label":"倒地姿态持续","value":"25 秒"},
                 {"key":"motion_drop","label":"倒地前运动变化","value":"突然下降"},
                 {"key":"shout_detected","label":"呼救语音检测","value":"未检测到"}
               ],
               "disclaimer":"AI 辅助判断，不替代医疗急救专业结论，请以人工确认为准"},
             "notifications":[
               {"target":"张伟","channel":"push+sms","sent_at":"2026-08-04T15:02:13+08:00","ack":false},
               {"target":"张莉","channel":"push","sent_at":"2026-08-04T15:02:13+08:00","ack":false}
             ],
             "escalation":{"auto_call_at":"2026-08-04T15:03:11+08:00","status":"pending"}}"""

        private const val CONTACTS_JSON = """
            {"contacts":[
              {"order":1,"name":"张伟","relation":"son","phone":"138****6688","channels":["push","sms","call"]},
              {"order":2,"name":"张莉","relation":"daughter","phone":"139****2233","channels":["push","sms"]},
              {"order":3,"name":"刘姐（社区网格员）","relation":"community","phone":"0571-****120","channels":["call"]}
            ]}"""

        private const val ELDERS_JSON = """
            {"elders":[
              {"elder_id":"u-elder-001","name":"王秀兰","relation":"son","overall":"danger",
               "last_active_at":"2026-08-04T16:33:00+08:00","pending_event_count":1}
            ]}"""

        private const val STATS_EVENTS_JSON = """
            {"buckets":[
              {"period":"第1周","reminder":1,"warning":1,"emergency":0},
              {"period":"第2周","reminder":2,"warning":0,"emergency":1},
              {"period":"第3周","reminder":1,"warning":0,"emergency":0},
              {"period":"第4周","reminder":3,"warning":2,"emergency":1}
            ]}"""

        private const val STATS_ACTIVITY_JSON = """
            {"hours":[0.05,0.03,0.02,0.04,0.08,0.15,0.35,0.6,0.85,1.0,0.9,0.55,
                      0.35,0.2,0.15,0.3,0.55,0.8,0.9,0.7,0.5,0.3,0.12,0.06]}"""
    }
}
