"""Версионируемые учебные критерии; смена формулировок требует новой оценки.

Критерии на английском не означают проверенную точность на русском.
Возврат денег, SLA и применение санкций не входят в полномочия классификатора.
"""

REFUND_QUESTIONS = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request money back or reversal of a charge?",
        "criteria": {
            "false": "No explicit request for money back; a billing question, complaint, or invoice request alone is not a refund request.",
            "true": "The customer explicitly asks for a refund, money back, or reversal of a payment or duplicate charge. This does not establish eligibility.",
        },
    },
}

ROUTE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Choose the department best suited to the primary request. Use other when the request is unclear or outside the listed departments.",
        "criteria": {
            "billing": "Existing invoices, payments, refunds, or duplicate charges.",
            "technical": "Product bugs, outages, integration errors, or broken functionality.",
            "sales": "Pre-purchase pricing, a new contract, plan comparison, or purchasing an upgrade.",
            "other": "An unclear request, general feedback, or a request not covered by the other departments.",
        },
    },
}

MODERATION_QUESTIONS = {
    "targeted_abuse": {
        "type": "noul",
        "instructions": "Does the author's own message contain targeted abuse or a threat against a person or group?",
        "criteria": {
            "false": "No targeted abuse or threat; respectful disagreement, criticism of a product, and reporting or condemning someone else's abuse do not qualify.",
            "true": "The author directly insults, degrades, harasses, or threatens a person or group; not merely a quotation reported for help.",
        },
    },
    "fraud_or_credential_theft": {
        "type": "noul",
        "instructions": "Does the author solicit secret credentials or promote a fraudulent payment scheme?",
        "criteria": {
            "false": "No credential theft or fraudulent solicitation; an ordinary payment question or a report warning about a scam does not qualify.",
            "true": "The author asks someone to disclose a password, one-time login code, or other secret credential, or promotes a deceptive money-transfer scheme.",
        },
    },
}

PRIORITY_QUESTIONS = {
    "priority": {
        "type": "score",
        "instructions": "Rate support urgency from the stated operational impact, not the author's tone. Do not invent impact or deadlines absent from the message.",
        "criteria": [
            "Low: informational question, optional improvement, or explicitly no operational impact or deadline.",
            "Normal: a non-blocking problem affecting limited work; a workaround is available and no immediate deadline is stated.",
            "High: important work is blocked for one user or a limited team, or a stated near-term deadline is at risk.",
            "Critical: a production-wide outage, active data loss, or an explicitly stated immediate safety or security incident.",
        ],
    },
}
