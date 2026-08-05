from __future__ import annotations

EVIDENCE_LABELS = {
    "identity_claim": {
        "stage": "contact",
        "strength": "weak",
        "description": "冒充机构、客服、熟人或权威身份",
    },
    "benefit_lure": {
        "stage": "contact",
        "strength": "weak",
        "description": "以退款、补贴、投资或保健优惠建立接触",
    },
    "emergency_pretext": {
        "stage": "contact",
        "strength": "medium",
        "description": "以亲属事故、被抓或抢救制造紧急情境",
    },
    "amount_request": {
        "stage": "probing",
        "strength": "medium",
        "description": "围绕具体金额或资产数量进行试探",
    },
    "sensitive_info_request": {
        "stage": "probing",
        "strength": "medium",
        "description": "索取身份证、银行卡或账户资料",
    },
    "credential_request": {
        "stage": "action",
        "strength": "strong",
        "description": "索取验证码、密码或账户凭证",
    },
    "remote_control_instruction": {
        "stage": "action",
        "strength": "strong",
        "description": "要求共享屏幕、安装软件或接受远程控制",
    },
    "money_instruction": {
        "stage": "action",
        "strength": "strong",
        "description": "要求转账、汇款、取现、充值或付款",
    },
    "secrecy_control": {
        "stage": "control",
        "strength": "strong",
        "description": "要求对家人保密、不报警或不中断联系",
    },
    "urgency_pressure": {
        "stage": "control",
        "strength": "weak",
        "description": "通过时限、威胁或连续催促施压",
    },
    "protective_warning": {
        "stage": "protective",
        "strength": "protective",
        "description": "反诈提醒、劝阻转账或保护性否定表达",
    },
}

CLASSIFIER_LABELS = tuple(EVIDENCE_LABELS)
