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
            path == "/api/v1/auth/login" && method == "POST" -> ok(LOGIN_JSON)
            path == "/api/v1/auth/logout" && method == "POST" -> ok("{}")
            path == "/api/v1/users/me" -> ok(USER_JSON)

            // ---------- 事件 ----------
            path == "/api/v1/events" && method == "GET" -> ok(EVENTS_JSON)
            path.matches(Regex("/api/v1/events/[^/]+")) && method == "GET" -> ok(DETAIL_JSON)
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
        private const val LOGIN_JSON = """
            {"access_token":"mock-access-token","token_type":"bearer",
             "expires_at":"2026-08-13T10:00:00+08:00",
             "user":{"user_id":"u-family-001","role":"family","name":"演示家属"}}"""

        private const val USER_JSON = """
            {"user_id":"u-family-001","role":"family","name":"演示家属",
             "bound_family_count":0,"font_size":"large","voice_assist_enabled":false}"""

        private const val DEVICES_JSON = """
            {"devices":[
              {"device_id":"camera-01","name":"客厅摄像头","room":"living_room","online":true,"signal":"good","last_seen_at":"2026-08-04T16:44:58+08:00"},
              {"device_id":"camera-02","name":"卧室摄像头","room":"bedroom","online":true,"signal":"good","last_seen_at":"2026-08-04T16:44:58+08:00"},
              {"device_id":"camera-03","name":"厨房摄像头","room":"kitchen","online":true,"signal":"weak","last_seen_at":"2026-08-04T16:40:12+08:00"},
              {"device_id":"camera-04","name":"门口摄像头","room":"door","online":false,"signal":"offline","last_seen_at":"2026-08-04T08:10:00+08:00"}
            ]}"""

        private const val EVENTS_JSON = """
            {"events":[
              {"event_id":"evt_002","type":"fraud_suspected","level":"warning","title":"疑似诈骗电话",
               "summary":"来电自称“银行客服”并索要短信验证码，风险等级 S4","device_id":"camera-01",
               "occurred_at":"2026-08-04T11:20:05+08:00","status":"open","evidence_image_url":null,
               "fraud_scene":"telecom","fraud_state":"S4_ACTION_INDUCEMENT","fraud_state_index":4,
               "fraud_state_label":"敏感操作诱导","fraud_decision":"block"},
              {"event_id":"evt_006","type":"fraud_suspected","level":"emergency","title":"疑似入户诈骗",
               "summary":"访客停留期间出现保健品付款与保密诱导","device_id":"camera-04",
               "occurred_at":"2026-08-03T16:48:00+08:00","status":"resolved","evidence_image_url":null,
               "fraud_scene":"home_visit","fraud_state":"S5_HIGH_RISK_CONTROL","fraud_state_index":5,
               "fraud_state_label":"高危控制与执行","fraud_decision":"intervene"}
            ],"next_cursor":null}"""

        private const val DETAIL_JSON = """
            {"event_id":"evt_002","type":"fraud_suspected","level":"warning","status":"open",
             "device_id":"camera-01","occurred_at":"2026-08-04T11:20:05+08:00",
             "evidence_image_url":null,
             "analysis":{"confidence":0.91,
               "reasons":[
                 {"key":"phone_call_active","label":"通话场景","value":"检测到持续通话"},
                 {"key":"identity_claim","label":"身份冒充","value":"自称银行客服"},
                 {"key":"credential_request","label":"敏感信息","value":"索要短信验证码"}
               ],
               "disclaimer":"AI 辅助判断，不替代公安、银行或支付机构的专业结论"},
             "notifications":[],
             "escalation":{"auto_call_at":null,"status":"pending"},
             "fraud":{"scene":"telecom","state":"S4_ACTION_INDUCEMENT","state_index":4,
               "state_label":"敏感操作诱导","decision":"block","transition_reason":"出现验证码索取和身份冒充证据"}}"""

        private const val CONTACTS_JSON = """
            {"contacts":[
              {"order":1,"name":"家属一","relation":"son","phone":"138****6688","channels":["push","sms","call"]},
              {"order":2,"name":"家属二","relation":"daughter","phone":"139****2233","channels":["push","sms"]},
              {"order":3,"name":"社区联系人","relation":"community","phone":"0571-****120","channels":["call"]}
            ]}"""

        private const val ELDERS_JSON = """
            {"elders":[
              {"elder_id":"u-elder-001","name":"演示老人","relation":"son","overall":"danger",
               "last_active_at":"2026-08-04T16:33:00+08:00","pending_event_count":1}
            ]}"""

    }
}
