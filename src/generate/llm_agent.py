"""GenAI red-team agent (Anthropic Claude) — the generative layer of Pillar 2.

Three jobs, in increasing order of how much they matter:

  1. generate_attack_artifact(): produce a realistic *content* artifact for an
     attack (phishing copy, vishing script, synthetic-identity dossier, dispute
     narrative, ...). These populate the UI's fidelity view and train the text
     classifier. Purpose is DEFENSIVE: synthetic examples to detect fraud.
  2. propose_attack_spec(): the important one. Given MEASURED evidence about what
     the current defense relies on — which signals carry its ranking of a family,
     and what the transactions that escaped it look like — propose the next
     attack generation as a STRUCTURED SPECIFICATION: which behavioural dial to
     move, in which direction, and which detector signal that is meant to defeat.
  3. propose_evasion(): the older scalar interface, kept so existing callers and
     saved artifacts keep working.

The model never writes transaction rows. It writes a specification, which is put
through the payment-domain constraint layer in ``attack_spec.validate_spec`` and
only then executed by the deterministic simulator. That split is what keeps the
system both creative and reproducible: swap Claude for the offline heuristic and
the pipeline still runs, byte-for-byte, from the same seed.

Everything degrades gracefully offline: if no API key and nothing cached, we
fall back to deterministic templates and to the weakness-driven heuristic, so the
pipeline, tests, and web app always run. Live calls are cached to disk by prompt
hash for reproducibility.
"""
from __future__ import annotations

import json

import numpy as np

import config
from src.generate.attack_spec import (AttackSpec, BASE_SPECS, SpecRejected,
                                      validate_spec)
from src.identify.taxonomy import Attack, load_taxonomy
from src.llm.client import LLMClient, LLMUnavailable

ATTACKER_SYSTEM = (
    "You are a payment-fraud RED-TEAM simulator used to build DEFENSIVE training "
    "data for a bank's fraud-detection models. You produce synthetic, clearly "
    "fictional artifacts that mimic the *style and red flags* of a fraud attack so "
    "classifiers can learn to catch them. Never include operational instructions, "
    "real targets, working infrastructure, or real personal data. Focus on the "
    "observable linguistic and behavioural cues a detector should flag."
)

_ARTIFACT_TYPE = {
    "Social Engineering": "phishing_or_script",
    "Authorized Push Payment Scams": "persuasion_script",
    "Account & Identity Compromise": "synthetic_identity_dossier",
    "Merchant & Ecosystem Abuse": "dispute_or_merchant_narrative",
    "Payment Instrument Attacks": "attacker_playbook_note",
    "Adversarial ML & Model Evasion": "attacker_playbook_note",
}


def _artifact_prompt(attack: Attack) -> str:
    return (
        f"Attack: {attack.name}\nCategory: {attack.category}\n"
        f"Mechanism: {attack.genai_mechanism}\n"
        f"Red flags a detector should catch: {', '.join(attack.observable_signals)}\n\n"
        "Produce ONE synthetic, fictional artifact a fraud-detection team could add "
        "to a training set. Return JSON with keys: artifact_type (short string), "
        "subject (short string), body (2-5 sentences of the fictional lure/script/"
        "narrative, clearly synthetic), red_flags (array of 3-6 short strings)."
    )


def generate_attack_artifact(attack: Attack, client: LLMClient | None = None,
                             rng: np.random.Generator | None = None) -> dict:
    client = client or LLMClient()
    rng = rng or np.random.default_rng(config.GLOBAL_SEED)
    prompt = _artifact_prompt(attack)
    # 1) cache/live
    try:
        data = client.structured(prompt, ATTACKER_SYSTEM, schema_hint="object")
        if isinstance(data, dict) and data.get("body"):
            data.setdefault("artifact_type", _ARTIFACT_TYPE.get(attack.category, "note"))
            data["attack_id"] = attack.id
            data["source"] = "llm"
            return data
    except (LLMUnavailable, ValueError):
        pass
    # 2) offline template fallback
    return _template_artifact(attack, rng)


# Slot vocabulary for offline lure composition. Every slot is generic: no real
# brand, link, phone number, app name, or step-by-step instruction appears here.
# These are the *shapes* of scam messages a classifier needs to recognize.
_OPENERS = [
    "Dear customer,", "Important notice:", "Hello,", "Attention:", "Namaste,",
    "Customer service update:", "Final notice:", "Alert:",
]
_PRETEXTS = {
    "card_block": "we have detected a problem with your card and it will be blocked today",
    "kyc": "your KYC record has expired and your account will be frozen within 24 hours",
    "refund": "a refund is pending on your account but the payout details could not be verified",
    "parcel": "a parcel addressed to you is held pending a small customs charge",
    "reward": "you have been selected for a limited cashback offer on your last payment",
    "investment": "your portfolio has returned well above the guaranteed rate this quarter",
    "job": "your application for flexible part-time work has been approved",
    "emergency": "there has been an emergency and money is needed immediately",
    "invoice": "our banking details have changed and the outstanding invoice must be reissued",
    "support": "our support team has been assigned to resolve the issue reported on your account",
    "safe_account": "your account has been compromised and your balance is at risk",
    "mandate": "an automatic payment instruction on your account could not be processed",
}
_ACTIONS = [
    "verify your details on the page we have sent you",
    "read back the code that has just been sent to your phone",
    "approve the request that will appear on your device now",
    "transfer the balance to the protected account we have opened in your name",
    "pay a small verification amount so the payout can be released",
    "confirm the card number and expiry so we can release the hold",
    "install the assistance tool so our agent can complete this with you",
    "scan the code we are sending to collect the amount",
    "authorise the payment to the updated account details below",
]
_URGENCY = [
    "This must be completed within 30 minutes.",
    "Failure to act today will result in permanent suspension.",
    "Our agent will remain on the line until it is done.",
    "This reference expires shortly.",
    "Please do not discuss this with anyone until it is settled.",
    "Only one attempt is permitted before the account is locked.",
]

# Which pretexts fit which attack, so the corpus is varied but still coherent.
_PRETEXT_BY_KEYWORD = [
    ("vishing", ["card_block", "safe_account", "support"]),
    ("smishing", ["kyc", "reward", "parcel", "card_block"]),
    ("phishing", ["card_block", "kyc", "refund"]),
    ("support", ["support", "refund", "card_block"]),
    ("romance", ["investment", "emergency"]),
    ("investment", ["investment", "reward"]),
    ("invoice", ["invoice"]),
    ("bec", ["invoice", "emergency"]),
    ("marketplace", ["refund", "parcel"]),
    ("family", ["emergency"]),
    ("task", ["job", "investment"]),
    ("refund", ["refund", "support"]),
    ("mfa", ["safe_account", "card_block"]),
    ("sim_swap", ["kyc", "safe_account"]),
    ("kba", ["kyc", "support"]),
    ("mandate", ["mandate"]),
    ("wallet", ["card_block", "kyc"]),
    ("qr", ["reward", "refund"]),
    ("mule", ["job"]),
]


def _pretexts_for(attack: Attack) -> list[str]:
    for kw, themes in _PRETEXT_BY_KEYWORD:
        if kw in attack.id or kw in attack.name.lower():
            return themes
    if attack.category == "Authorized Push Payment Scams":
        return ["safe_account", "invoice", "investment"]
    return ["card_block", "kyc", "support"]


def _template_artifact(attack: Attack, rng: np.random.Generator) -> dict:
    """Deterministic offline lure, composed from slot vocabulary chosen by this
    attack's own catalog entry.

    Composition matters for more than variety. If every fraudulent sample shared
    a handful of sentences — or if the fraudulent samples were *descriptions* of
    attacks while the benign samples were real messages — a text classifier would
    separate the classes perfectly by genre and the reported score would mean
    nothing. Both classes here are short customer-facing messages.

    Nothing in this vocabulary is operational: no brand, link, number, app or
    procedure. These are the shapes of scam messages, which is what a scam-text
    detector has to learn.
    """
    flags = attack.observable_signals[:5]
    themes = _pretexts_for(attack)
    theme = themes[int(rng.integers(len(themes)))]
    opener = _OPENERS[int(rng.integers(len(_OPENERS)))]
    action = _ACTIONS[int(rng.integers(len(_ACTIONS)))]
    urgency = _URGENCY[int(rng.integers(len(_URGENCY)))]
    amount = int(rng.integers(199, 25000))
    pre = _PRETEXTS[theme]
    # Several sentence shapes, so no single connective or framing phrase appears
    # in every fraudulent sample. A shared scaffold is a label in disguise.
    patterns = [
        f"{opener} {pre}. To resolve this, {action}. {urgency}",
        f"{opener} {pre}. {urgency} Please {action}.",
        f"{pre.capitalize()} — {action} to continue. INR {amount} is involved. {urgency}",
        f"{opener} we are contacting you because {pre}. {urgency} You will need to {action}.",
        f"{urgency} {pre.capitalize()}. {action.capitalize()} and the matter will be closed.",
        f"{opener} regarding INR {amount} on your account: {pre}. Kindly {action}.",
    ]
    body = patterns[int(rng.integers(len(patterns)))]
    return {
        "artifact_type": _ARTIFACT_TYPE.get(attack.category, "note"),
        "subject": f"[SIM] {attack.name}",
        "body": f"[SYNTHETIC] {body}",
        "red_flags": flags,
        "attack_id": attack.id,
        "source": "template",
    }


_MARKERS = ("[SYNTHETIC]", "[SIM]", "[synthetic]", "[sim]")


def strip_markers(text: str) -> str:
    """Remove the provenance markers before the text is modelled.

    Every generated fraud artifact is tagged so a human reader can never mistake
    it for a real message. Those tags must not reach the classifier: a model that
    learns "contains the word SYNTHETIC" would score perfectly and detect nothing.
    """
    for m in _MARKERS:
        text = text.replace(m, " ")
    return " ".join(text.split())


def build_text_corpus(client: LLMClient | None = None, seed: int = config.GLOBAL_SEED,
                      n_benign: int = 140) -> list[dict]:
    """Fraud artifacts (label 1) + benign transactional and security messages
    (label 0) for the text classifier. Fully offline-capable.

    ``text`` is what the classifier sees, with provenance markers stripped;
    ``display_text`` keeps them so the UI still shows clearly-labelled synthetic
    content.
    """
    rng = np.random.default_rng(seed)
    tax = load_taxonomy()
    corpus: list[dict] = []
    # Only attacks that actually reach a victim as a MESSAGE belong in a
    # scam-text corpus. Including transaction-only attacks would mean the
    # fraudulent class was written in a different genre from the benign class,
    # and the classifier would be separating genres rather than intent.
    message_attacks = [a for a in tax.attacks
                       if a.category in ("Social Engineering", "Authorized Push Payment Scams")
                       or a.channel in ("MESSAGING", "VOICE")
                       or a.genai_role in ("content_generation", "multimodal_impersonation")]
    for a in message_attacks:
        for _ in range(4):
            art = generate_attack_artifact(a, client=client, rng=rng)
            display = f"{art.get('subject','')} {art.get('body','')}".strip()
            # The classifier sees the message BODY only. The subject line carries
            # the attack's catalog name for the UI, and leaving it in would let
            # the model read the label straight off the text.
            corpus.append({"text": strip_markers(art.get("body", "")),
                           "display_text": display,
                           "label": 1, "attack_id": a.id,
                           "artifact_type": art.get("artifact_type"),
                           "red_flags": art.get("red_flags", []),
                           "source": art.get("source")})
    corpus.extend(_benign_messages(rng, n_benign))
    rng.shuffle(corpus)
    return corpus


_BENIGN_TEMPLATES = [
    "Your payment of INR {amt} to {m} was successful. Thank you for shopping with us.",
    "Reminder: your {m} bill of INR {amt} is due on the 15th. Pay via the app.",
    "OTP for your {m} purchase is 4821. Do not share it with anyone.",
    "Order confirmed! Your {m} order will be delivered by Friday.",
    "Welcome to {m} rewards — you earned 120 points on your last purchase.",
    "Your card ending 4417 was used for INR {amt} at {m}. Not you? Call us.",
    "Statement ready: your {m} account summary for August is now available.",
    "Thanks for your subscription to {m}. Your next renewal is on the 1st.",
    # The genuinely confusable class: real security notices share the urgency,
    # authentication and money vocabulary that scam messages use. Without these
    # the classifier only has to learn "does this mention an OTP", which would be
    # a meaningless result.
    "Security alert: a new device signed in to your {m} account. If this was not "
    "you, secure your account from the app immediately.",
    "We have temporarily blocked a suspicious transaction of INR {amt} on your card. "
    "Review it in the app to confirm or reject.",
    "Action required: your KYC documents expire this month. Update them from the "
    "official {m} app — we will never ask for your OTP or PIN.",
    "Your request to add a new payee has been received. It will be active in 30 "
    "minutes as a cooling-off measure.",
    "Urgent: your {m} autopay mandate of INR {amt} failed. Retry from the app to "
    "avoid a service interruption.",
    "Verification complete. Your card has been added to your wallet on this device.",
    "Your refund of INR {amt} from {m} has been processed and will reach your account "
    "in 3 working days.",
    "Final reminder: your credit card payment of INR {amt} is due today. Pay now to "
    "avoid late charges.",
]
_BENIGN_MERCHANTS = ["BigBazaar", "Swiggy", "Amazon", "IRCTC", "Croma", "Tata Neu",
                     "Reliance Digital", "BookMyShow", "Zomato", "Flipkart"]


def _benign_messages(rng: np.random.Generator, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        tmpl = _BENIGN_TEMPLATES[int(rng.integers(len(_BENIGN_TEMPLATES)))]
        m = _BENIGN_MERCHANTS[int(rng.integers(len(_BENIGN_MERCHANTS)))]
        amt = int(rng.integers(120, 12000))
        text = tmpl.format(amt=amt, m=m)
        # Genuine notices also open formally and also press for action, so the
        # opener and the urgency vocabulary cannot be a shortcut to the label.
        if rng.random() < 0.45:
            text = f"{_OPENERS[int(rng.integers(len(_OPENERS)))]} {text[0].lower()}{text[1:]}"
        if rng.random() < 0.30:
            text = f"{text} {_URGENCY[int(rng.integers(len(_URGENCY)))]}"
        out.append({"text": text, "display_text": text, "label": 0,
                    "attack_id": "legit", "artifact_type": "transactional",
                    "red_flags": [], "source": "template"})
    return out


REDTEAM_SPEC_SYSTEM = (
    "You are the red-team strategist inside a bank's DEFENSIVE fraud-simulation "
    "laboratory. A detection model has just been stress-tested against simulated "
    "attacks, and you are given measurements of which signals that model relies on. "
    "Your job is to propose the next generation of the SIMULATED attack so the "
    "defense can be trained against it before real fraud gets there. "
    "You never produce operational guidance, tooling, or instructions anyone could "
    "act on: you only choose values on a fixed set of behavioural dials that a "
    "sandboxed simulator will execute. Everything you describe stays inside that "
    "simulator."
)

_SPEC_VOCAB = (
    "amount_profile: micro | low | moderate | high | extreme\n"
    "velocity_profile: single | low_and_slow | moderate | burst\n"
    "device_behavior: trusted_device | secondary_device | new_device | shared_device\n"
    "geo_behavior: home | plausible | domestic_far | foreign | high_risk\n"
    "merchant_behavior: known_merchant | new_low_risk_merchant | new_high_risk_merchant | "
    "front_merchant | cash_like\n"
    "timing_profile: customer_normal | business_hours | night | any\n"
    "intensity: number between 0.15 and 1.0, must be <= the current intensity"
)


def _spec_prompt(family: str, previous: AttackSpec, report) -> str:
    relied = ", ".join(f"{s['feature']} (AUC drop {s['auc_drop']:.3f})"
                       for s in report.relied_signals[:5]) or "not measured"
    escaped = "; ".join(
        f"{p['feature']}: escaped={p['escaped_mean']:.2f} vs caught={p['caught_mean']:.2f} "
        f"vs genuine={p['legit_mean']:.2f}" for p in report.escape_profile[:5]) or "none escaped"
    return (
        f"Attack family under test: {family}\n"
        f"Current specification: {json.dumps(previous.to_dict()['strategy'])} with "
        f"amount={previous.amount_profile}, velocity={previous.velocity_profile}, "
        f"device={previous.device_behavior}, geo={previous.geo_behavior}, "
        f"merchant={previous.merchant_behavior}, timing={previous.timing_profile}, "
        f"intensity={previous.intensity:.2f}\n\n"
        f"MEASURED DEFENSE BEHAVIOUR\n"
        f"- recall on this family: {report.recall if report.recall is not None else 'n/a'}\n"
        f"- transactions of this family that escaped: {report.n_escaped} of {report.n_fraud}\n"
        f"- signals the model's ranking of this family depends on: {relied}\n"
        f"- how escaped transactions differ from caught ones: {escaped}\n\n"
        "Propose the NEXT generation of this simulated attack. Change as few dials as "
        "possible — ideally one — and change the one that removes the model's strongest "
        "dependency, so that whatever the defense still catches afterwards is a genuinely "
        "different signal rather than the same one.\n\n"
        f"Permitted values:\n{_SPEC_VOCAB}\n\n"
        "Return JSON with keys: strategy (one sentence), amount_profile, velocity_profile, "
        "device_behavior, geo_behavior, merchant_behavior, timing_profile, intensity, "
        "targets_signal (the detector signal this generation is meant to defeat), "
        "rationale (one sentence), confidence (0-1)."
    )


def propose_attack_spec(family: str, report, previous: AttackSpec | None = None,
                        client: LLMClient | None = None):
    """Propose the next attack generation as a validated :class:`AttackSpec`.

    Returns ``(spec, validation_report_dict)``. The language model proposes; the
    payment-domain constraint layer disposes. If the model is unavailable, or
    returns something unusable, the weakness-driven offline heuristic produces
    the proposal instead — the loop's behaviour degrades in quality, never in
    reproducibility.
    """
    from src.loop.weakness import heuristic_mutation

    previous = previous or BASE_SPECS[family]
    fallback = heuristic_mutation(report, previous)

    client = client or LLMClient()
    try:
        data = client.structured(_spec_prompt(family, previous, report),
                                 REDTEAM_SPEC_SYSTEM, schema_hint="object")
        if not isinstance(data, dict) or not data:
            raise ValueError("empty specification")
        data["source"] = "llm"
        data.setdefault("generation", previous.generation + 1)
        spec, validation = validate_spec(data, family, previous=previous)
        return spec, validation.to_dict()
    except (LLMUnavailable, ValueError, TypeError, KeyError, SpecRejected):
        spec, validation = validate_spec(fallback, family, previous=previous)
        return spec, validation.to_dict()


def propose_evasion(attack_type: str, recall: float, current_intensity: float = 1.0,
                    client: LLMClient | None = None) -> dict:
    """Given an attack family, propose stealthier params for the NEXT wave.

    `intensity` is the attack's aggressiveness: 1.0 = blatant, 0.2 = nearly
    indistinguishable from legitimate behaviour. A persistent attacker only ever
    escalates stealth, so the returned intensity is always <= current_intensity.
    Returns {"intensity", "tactic", "rationale", "source"}. Offline heuristic:
    push harder when the defense is strong (high recall), ease off slightly when
    already evading (low recall), but never regress on stealth."""
    client = client or LLMClient()
    tax = load_taxonomy()
    attacks = tax.for_injector(attack_type)
    signals = sorted({s for a in attacks for s in a.observable_signals})
    prompt = (
        f"A fraud classifier detects the '{attack_type}' attack family with recall "
        f"{recall:.2f}; the current attack aggressiveness is {current_intensity:.2f} "
        "(1.0 = blatant, 0.2 = near-legitimate). Detectable signals it relies on: "
        f"{signals}. Propose how the red-team should make the NEXT wave stealthier "
        "(for defensive stress-testing). Return JSON: intensity (number 0.2-1.0, "
        "<= current, lower = closer to legitimate), tactic (short string), rationale "
        "(one sentence)."
    )
    try:
        data = client.structured(prompt, ATTACKER_SYSTEM, schema_hint="object")
        intensity = float(data.get("intensity", current_intensity * 0.7))
        intensity = float(min(current_intensity, max(0.2, intensity)))
        return {"intensity": intensity,
                "tactic": str(data.get("tactic", "reduce deviation from baseline")),
                "rationale": str(data.get("rationale", "")), "source": "llm"}
    except (LLMUnavailable, ValueError, TypeError):
        factor = 0.6 if recall >= 0.85 else 0.8   # push harder when defense is strong
        intensity = float(np.clip(current_intensity * factor, 0.2, current_intensity))
        return {"intensity": intensity,
                "tactic": "shrink amount/geo/device deviations toward the cardholder centroid",
                "rationale": "Move fraudulent transactions closer to legitimate behaviour to evade the current boundary.",
                "source": "heuristic"}
