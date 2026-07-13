def calculate_status(risk_score):

    if risk_score >= 70:
        return "Critical"

    elif risk_score >= 30:
        return "Warning"

    return "Healthy"


def calculate_health_score(risk_score):

    return round(100 - risk_score, 1)


def maintenance_recommendation(risk_score):

    if risk_score >= 70:

        return (
            "Immediate maintenance required. "
            "Inspect cutting tool, cooling system "
            "and reduce machine load."
        )

    elif risk_score >= 30:

        return (
            "Machine should be monitored. "
            "Schedule preventive maintenance."
        )

    return (
        "Machine is operating normally. "
        "Continue routine maintenance."
    )