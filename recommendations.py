def get_advice(class_name: str) -> dict:
    """General educational guidance. Verify disease and treatment locally."""
    label = class_name.lower()

    if "healthy" in label:
        return {
            "summary": "The model classified this image as a healthy leaf.",
            "steps": [
                "Continue regular field monitoring.",
                "Water according to crop and soil needs; avoid waterlogging.",
                "Maintain suitable spacing and remove weeds.",
                "Recheck plants regularly for new spots, wilting or discoloration.",
            ],
            "disclaimer": "A healthy prediction does not guarantee the whole plant is disease-free.",
        }

    if "early_blight" in label:
        steps = [
            "Remove badly affected leaves where practical and dispose of them away from the crop.",
            "Avoid overhead irrigation and keep foliage as dry as practical.",
            "Improve airflow through appropriate plant spacing and pruning.",
            "Rotate crops and remove volunteer host plants where recommended locally.",
        ]
    elif "late_blight" in label:
        steps = [
            "Isolate and seek prompt local agricultural advice; this disease can spread rapidly in suitable conditions.",
            "Avoid handling wet plants and avoid moving potentially infected plant material through the field.",
            "Remove infected material only using locally recommended sanitation practices.",
            "Ask an agricultural extension officer about locally approved management options.",
        ]
    elif "bacterial" in label:
        steps = [
            "Avoid working among plants when foliage is wet.",
            "Clean tools between plants and avoid unnecessary leaf injury.",
            "Use healthy planting material and follow locally recommended crop rotation.",
            "Seek local confirmation because bacterial and fungal symptoms can look similar.",
        ]
    elif "leaf_mold" in label or "mold" in label:
        steps = [
            "Improve ventilation and reduce prolonged leaf wetness.",
            "Avoid excessive irrigation and manage humidity where possible.",
            "Remove severely affected leaves using clean tools.",
            "Confirm the cause locally before choosing any treatment.",
        ]
    elif "mosaic" in label or "yellow" in label or "curl" in label:
        steps = [
            "Check for insect vectors such as aphids or whiteflies and seek local integrated pest-management advice.",
            "Remove severely affected plants only after confirming the diagnosis and local guidance.",
            "Control weeds that may host pests or pathogens.",
            "Use clean planting material and monitor nearby plants.",
        ]
    else:
        steps = [
            "Take additional photos of the whole plant and affected leaves in natural light.",
            "Check whether symptoms are spreading and note recent weather and irrigation.",
            "Remove severely damaged material only if local agricultural guidance recommends it.",
            "Consult a local agricultural expert to confirm the disease before applying treatment.",
        ]

    return {
        "summary": "The model detected a class associated with a possible plant health issue. Treat this as an initial screening result, not a confirmed diagnosis.",
        "steps": steps,
        "disclaimer": "Recommendations are general. Confirm the diagnosis and use only locally approved treatments at label directions. Never mix or apply chemicals based only on an AI result.",
    }
