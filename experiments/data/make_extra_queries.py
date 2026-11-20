"""Generates the EXTRA labelled queries (the 4x set) for the three long policies.

    python experiments/data/make_extra_queries.py

Each base question is defined once, compactly: four wordings and one or more FACT GROUPS. For a group:
    sections  every section whose text states the fact (the primary one first)
    strings   distinctive text a chunk must contain to count as holding the fact (boundary-aware matching)
    answer    phrases an answer may use to state the fact
The golden text is copied from the policy itself (whole section body, or the table header + the named row), so
it is verbatim by construction. Output: <company>_<version>_queries_extra.jsonl and _facts_extra.json.
Labels are then audited by experiments/validate_labels.py --extra.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent
VERSIONS = {"acme": "v2", "globex": "v1", "initech": "v1"}
PROFILE = {"acme": {"grade": "L3", "shift": "general", "gender": "female"},
           "globex": {"grade": "L3", "shift": "general", "gender": "female"},
           "initech": {"grade": "L2", "shift": "general", "gender": "female"}}
VARIANTS = ("original", "paraphrase-1", "paraphrase-2", "paraphrase-3")   # keyword, chatty, synonyms


def g(sections, strings, answer, summary, golden=None):
    """A fact group. golden = (section id, table row key) for a table row, else None for the primary section."""
    return {"sections": sections, "strings": strings, "answer": answer, "summary": summary, "golden": golden}


def base(category, questions, *groups):
    return {"category": category, "q": questions, "groups": list(groups)}


# ------------------------------------------------------------------------------------------------ definitions

ACME = [
    base("direct", ["What is the longest one-way journey time the transport desk aims to keep a route within?",
                    "Maximum one-way ride duration?",
                    "I live quite far out and my commute home takes forever. Is there a limit on how long a single trip on the company cab is allowed to take?",
                    "What is the upper bound on travel time for one leg of a shared pickup route?"],
         g(["12.1", "12.5", "25.1"], ["90 minutes", "90-minute"], ["90 minutes", "90-minute"], "The one-way journey should stay within 90 minutes.")),
    base("direct", ["How old can a vehicle be to be used for employee transport?",
                    "Vehicle age limit?",
                    "Our vendor sent a very old car to pick me up yesterday and it made me wonder whether there is any rule about how old the cabs can be.",
                    "Up to how many years after first registration may an allotted car stay in service?"],
         g(["13.1", "25.3"], ["five years", "5 years"], ["five years", "5 years"], "A vehicle may be at most 5 years from first registration.")),
    base("direct", ["How long is trip location data kept after a journey?",
                    "Location data retention period?",
                    "I was thinking about privacy, since the app tracks the cab I am in. How long do they hold on to the GPS history of my rides?",
                    "For what length of time are the recorded positions from a journey stored before being discarded?"],
         g(["19.3", "25.4"], ["90 calendar days"], ["90 calendar days", "90 days"], "Trip location data is retained for 90 calendar days.")),
    base("table", ["At what time does the cab leave the office for employees on the Early shift?",
                   "Early shift drop time?",
                   "I start work very early in the morning, around six, and want to know when the cab home actually departs at the end of that shift.",
                   "What is the departure time of the return vehicle for the earliest roster slot?"],
         g(["3.1"], ["3:15 PM"], ["3:15 PM"], "The Early shift drop departs at 3:15 PM.", golden=("3.1", "Early"))),
    base("table", ["What is the company-paid limit for a trip taken because I worked later than the last scheduled drop?",
                   "Late stay cab cap?",
                   "Sometimes I end up working well past the final scheduled cab and have to take a separate one home. How much will the company cover for that ride?",
                   "What is the maximum reimbursable amount for travel after the last timetabled departure?"],
         g(["8.2", "25.6"], ["Late stay beyond last drop"], ["₹1,000"], "A late-stay trip is capped at ₹1,000.", golden=("8.2", "Late stay"))),
    base("table", ["What is the per-trip cost cap for transporting a guest?",
                   "Guest trip cap?",
                   "We have a candidate visiting for interviews next week and I want to arrange her cab. Is there a ceiling on what the company will pay for a visitor's ride?",
                   "Up to what amount is a single journey for a visitor to the company funded?"],
         g(["8.2", "25.6"], ["₹3,500"], ["₹3,500"], "A guest trip is capped at ₹3,500.", golden=("8.2", "Guest"))),
    base("multi_section", ["How old can an allotted vehicle be, and how many hours may its driver work in a day?",
                           "Vehicle age limit and driver daily hours?",
                           "I am a bit worried about road safety on my route. Is there a cap on how old the car can be, and is there any limit on how long the driver is allowed to be on duty each day?",
                           "What are the ceilings on vehicle vintage and on a chauffeur's daily working time?"],
         g(["13.1", "25.3"], ["five years", "5 years"], ["five years", "5 years"], "A vehicle may be at most 5 years from first registration."),
         g(["13.3", "25.3", "25.5"], ["10 hours"], ["10 hours"], "A driver may work at most 10 hours of driving duty a day.")),
    base("multi_section", ["What is the limit on emergency medical expenses, and from which week of pregnancy can an employee ask for door-to-door transport?",
                           "Medical emergency cap and pregnancy door-to-door week?",
                           "A colleague of mine is expecting and also worries about paying for an emergency hospital trip. Which week does the home pickup service start, and how much is covered for emergencies?",
                           "What sum is covered for urgent health-related spending, and at what stage of pregnancy does doorstep collection become available?"],
         g(["14.3", "15.5", "18.5", "25.4"], ["₹2,500"], ["₹2,500"], "Emergency medical expenditure is capped at ₹2,500 per incident."),
         g(["15.2", "25.4"], ["24th week", "From week 24"], ["24th week", "week 24"], "Door-to-door service is available from the 24th week of pregnancy.")),
    base("multi_section", ["How long do I have to dispute a transport charge, and how quickly must the desk acknowledge a complaint?",
                           "Billing dispute deadline and complaint acknowledgement time?",
                           "I was charged for a trip I never took. How many days do I have to contest the charge, and once I complain, how soon will someone confirm they have received it?",
                           "Within what period must a charge be challenged, and what is the turnaround for confirming receipt of a grievance?"],
         g(["18.3", "25.5"], ["30 calendar days"], ["30 calendar days"], "A billing dispute must be raised within 30 calendar days."),
         g(["20.2", "25.5"], ["24 hours"], ["24 hours"], "A complaint is acknowledged within 24 hours.")),
    base("sibling", ["If I switch from coming to the office to working from home, how far ahead must I cancel my cab?",
                     "Hybrid day cab cancellation deadline?",
                     "Our team is moving to a hybrid schedule, so on some days I will work from home instead of coming in. How early before the pickup do I have to cancel the cab for those days?",
                     "What advance notice is needed to drop transport on a day I move from the workplace to remote work?"],
         g(["11.2", "25.2"], ["12 hours"], ["12 hours"], "A hybrid-day booking must be cancelled at least 12 hours before pickup.")),
    base("sibling", ["For how many days at a stretch can I book recurring transport?",
                     "Recurring booking length?",
                     "I travel the same route every working day, so I would rather not rebook each morning. What is the longest period I can set up in one go?",
                     "Over what span can a repeating transport arrangement be requested in a single submission?"],
         g(["4.3", "25.2"], ["Recurring booking horizon", "recurring transport for a period of up to 30 calendar days"], ["30 calendar days"],
           "Recurring transport may be requested for up to 30 calendar days at a time.")),
    base("sibling", ["What is the farthest I may be asked to walk to reach the boarding point?",
                     "Maximum walk to boarding point?",
                     "The assigned pickup spot is quite a trek from my building. Is there a limit on how far from my address the boarding point may be?",
                     "What is the greatest permitted pedestrian distance between my registered location and the stop where I join the cab?"],
         g(["12.2", "12.7", "22.5", "25.1"], ["500 metres", "500-metre"], ["500 metres", "500-metre"], "The walk to the boarding point must not exceed 500 metres.")),
]

GLOBEX = [
    base("direct", ["How far may the boarding point be from my home?",
                    "Maximum walk to boarding point?",
                    "The assigned pickup spot is quite a trek from my building. Is there a limit on how far from my address the boarding point may be?",
                    "What is the greatest permitted pedestrian distance between my registered location and the stop where I join the cab?"],
         g(["12.4", "25.1"], ["300 metres"], ["300 metres"], "The walk to the boarding point must not exceed 300 metres.")),
    base("direct", ["How old can a vehicle be to be used for employee transport?",
                    "Vehicle age limit?",
                    "Our vendor sent a very old car to pick me up yesterday and it made me wonder whether there is any rule about how old the cabs can be.",
                    "Up to how many years after original registration may an allotted car stay in service?"],
         g(["13.1", "25.1"], ["4 years", "four years"], ["4 years", "four years"], "A vehicle must be no more than 4 years old.")),
    base("direct", ["How long is journey location data kept?",
                    "Location data retention period?",
                    "I was thinking about privacy, since the app tracks the cab I am in. How long do they hold on to the GPS history of my rides?",
                    "For what length of time are the recorded positions from a journey stored before being discarded?"],
         g(["19.2", "25.5"], ["180 days"], ["180 days"], "Journey location data is retained for 180 days.")),
    base("table", ["At what time does the cab leave the office for employees on the Morning shift?",
                   "Morning shift drop time?",
                   "I start work very early in the morning, around seven, and want to know when the cab home actually departs at the end of that shift.",
                   "What is the departure time of the return vehicle for the earliest roster slot?"],
         g(["3.1"], ["4:15 PM"], ["4:15 PM"], "The Morning shift drop departs at 4:15 PM.", golden=("3.1", "Morning"))),
    base("table", ["What is the company-paid limit for a trip taken because I worked later than the last scheduled drop?",
                   "Late stay cab cap?",
                   "Sometimes I end up working well past the final scheduled cab and have to take a separate one home. How much will the company cover for that ride?",
                   "What is the maximum reimbursable amount for travel after the last timetabled departure?"],
         g(["8.2", "25.4"], ["₹600"], ["₹600"], "A late-stay trip is capped at ₹600.", golden=("8.2", "Late stay"))),
    base("table", ["What happens at the third no-show in a calendar month?",
                   "Third no-show consequence?",
                   "I have already missed my pickup twice this month because of late meetings. If it happens a third time, what will be done about it?",
                   "What action follows a third recorded failure to board within a single month?"],
         g(["7.1", "7.2", "25.3"], ["2 weeks", "two weeks"], ["2 weeks", "two weeks"], "A third no-show suspends transport for 2 weeks.", golden=("7.2", "| 3 |"))),
    base("multi_section", ["What is the longest allowed one-way journey, and how far may the walk to the boarding point be?",
                           "Maximum journey time and walking distance?",
                           "My commute is long and my pickup spot is far from my building. What are the limits on how long the trip may take and how far I can be asked to walk?",
                           "What are the ceilings on travel time for one leg of a route and on pedestrian distance to the stop?"],
         g(["6.3", "12.1", "12.5", "25.1"], ["75 minutes", "75-minute"], ["75 minutes", "75-minute"], "The one-way journey is limited to 75 minutes."),
         g(["12.4", "25.1"], ["300 metres"], ["300 metres"], "The walk to the boarding point must not exceed 300 metres.")),
    base("multi_section", ["What on-time pickup target must the vendor meet, and how quickly must it acknowledge a dispatch instruction?",
                           "Vendor on-time target and dispatch acknowledgement?",
                           "Our cab was late again. What punctuality level is the vendor supposed to reach, and how fast should it confirm that it has received a new dispatch order?",
                           "What performance level is demanded of the contractor for timely collection, and within what interval must it confirm a dispatch?"],
         g(["17.2", "25.2"], ["98%"], ["98%"], "The vendor's on-time pickup target is 98%."),
         g(["17.3", "25.2"], ["10 minutes"], ["10 minutes"], "A dispatch instruction must be acknowledged within 10 minutes.")),
    base("multi_section", ["How soon must an incident be reported, and how much is covered for emergency medical expenses?",
                           "Incident report deadline and emergency medical cap?",
                           "I witnessed an accident involving my cab. How quickly do I have to tell the desk, and is there a limit on what the company pays for urgent medical care?",
                           "Within what time must a transport incident be notified, and up to what sum is urgent health expenditure authorised?"],
         g(["14.2", "25.5"], ["1 hour"], ["1 hour", "one hour"], "An incident must be reported within 1 hour."),
         g(["8.2", "14.3", "25.4"], ["₹1,500"], ["₹1,500"], "Emergency medical expenditure is capped at ₹1,500 per incident.")),
    base("sibling", ["How early must I cancel my cab for a hybrid office day if I will be working from home instead?",
                     "Hybrid day cab cancellation deadline?",
                     "Our team is moving to a hybrid schedule, so on some days I will work from home instead of coming in. How early before the pickup do I have to cancel the cab for those days?",
                     "What advance notice is needed to drop transport on a day I move from the workplace to remote work?"],
         g(["11.2", "11.4", "25.2"], ["24 hours", "24-hour"], ["24 hours", "24-hour"], "A hybrid-day cancellation must be made at least 24 hours before pickup.")),
    base("sibling", ["How many days can an ordinary recurring transport request cover in one submission?",
                     "Recurring booking length?",
                     "I travel the same route every working day, so I would rather not rebook each morning. What is the longest period I can set up in one go?",
                     "Over what span can a repeating transport arrangement be requested in a single submission?"],
         g(["4.3", "25.2"], ["Recurring booking horizon", "no more than 14 calendar days"], ["14 calendar days"],
           "An ordinary recurring request may cover no more than 14 calendar days.")),
    base("sibling", ["How long after a complaint is closed can I ask for the outcome to be reviewed?",
                     "Complaint outcome review window?",
                     "My complaint was closed last week but I do not agree with how it was resolved. How long do I have to ask for someone to look at it again?",
                     "Within what period after closure may an employee request a review of a grievance decision?"],
         g(["20.7", "25.6"], ["Complaint outcome review request", "5 working days of receiving the closure communication"], ["5 working days"],
           "A review of the outcome may be requested within 5 working days of receiving the closure communication.")),
]

INITECH = [
    base("direct", ["What is the longest one-way journey time the transport desk aims to keep a route within?",
                    "Maximum one-way ride duration?",
                    "I live quite far out and my commute home takes forever. Is there a limit on how long a single trip on the company cab is allowed to take?",
                    "What is the upper bound on travel time for one leg of a shared pickup route?"],
         g(["6.4", "12.1", "12.5", "25.1"], ["120 minutes", "120-minute"], ["120 minutes", "120-minute"], "The one-way journey is limited to 120 minutes.")),
    base("direct", ["How long do I have to dispute a transport charge?",
                    "Billing dispute window?",
                    "I was charged for a trip I never took. How many days do I have to contest the charge before it is too late?",
                    "Within what period must a charge be challenged?"],
         g(["18.2", "25.4"], ["60 calendar days"], ["60 calendar days"], "A billing dispute must be raised within 60 calendar days.")),
    base("direct", ["How long is journey location data kept?",
                    "Location data retention period?",
                    "I was thinking about privacy, since the app tracks the cab I am in. How long do they hold on to the GPS history of my rides?",
                    "For what length of time are the recorded positions from a journey stored before being discarded?"],
         g(["19.2", "25.5"], ["30 days"], ["30 days"], "Journey location data is retained for 30 days.")),
    base("table", ["At what time does the cab leave the office for employees on the Early shift?",
                   "Early shift drop time?",
                   "I start work very early in the morning, around half past six, and want to know when the cab home actually departs at the end of that shift.",
                   "What is the departure time of the return vehicle for the earliest roster slot?"],
         g(["3.1"], ["3:45 PM"], ["3:45 PM"], "The Early shift drop departs at 3:45 PM.", golden=("3.1", "Early"))),
    base("table", ["What is the per-trip cost cap for a guest journey?",
                   "Guest trip cap?",
                   "We have a candidate visiting for interviews next week and I want to arrange her cab. Is there a ceiling on what the company will pay for a visitor's ride?",
                   "Up to what amount is a single journey for a visitor to the company funded?"],
         g(["8.2", "9.3", "25.4"], ["₹2,500"], ["₹2,500"], "A guest trip is capped at ₹2,500.", golden=("8.2", "Guest"))),
    base("table", ["What happens at the third no-show in a calendar month?",
                   "Third no-show consequence?",
                   "I have already missed my pickup twice this month because of late meetings. If it happens a third time, what will be done about it?",
                   "What action follows a third recorded failure to board within a single month?"],
         g(["7.2", "25.3"], ["1 month", "1-month"], ["1 month", "1-month"], "A third no-show suspends transport for 1 month.", golden=("7.2", "| 3 |"))),
    base("multi_section", ["How old can an allotted vehicle be, and how many hours may its driver work in a day?",
                           "Vehicle age limit and driver daily hours?",
                           "I am a bit worried about road safety on my route. Is there a cap on how old the car can be, and is there any limit on how long the driver is allowed to be on duty each day?",
                           "What are the ceilings on vehicle vintage and on a chauffeur's daily working time?"],
         g(["13.1", "25.1"], ["7 years", "seven years"], ["7 years", "seven years"], "A vehicle may be at most 7 years old."),
         g(["13.3", "25.1"], ["12 duty hours", "12 hours"], ["12 duty hours", "12 hours"], "A driver may work at most 12 duty hours in a day.")),
    base("multi_section", ["What on-time pickup target must the vendor meet, and how quickly must it acknowledge a dispatch instruction?",
                           "Vendor on-time target and dispatch acknowledgement?",
                           "Our cab was late again. What punctuality level is the vendor supposed to reach, and how fast should it confirm that it has received a new dispatch order?",
                           "What performance level is demanded of the contractor for timely collection, and within what interval must it confirm a dispatch?"],
         g(["17.3", "25.2"], ["90%"], ["90%"], "The vendor's on-time pickup target is 90%."),
         g(["17.4", "25.2"], ["10 minutes"], ["10 minutes"], "A dispatch instruction must be acknowledged within 10 minutes.")),
    base("multi_section", ["How quickly must the desk acknowledge a complaint, and how long after closure can I ask for a review?",
                           "Complaint acknowledgement time and appeal window?",
                           "I raised a complaint and want to know how soon someone must confirm they have it, and how long I have to challenge the result once it is closed.",
                           "Within what time must receipt of a grievance be confirmed, and how long after closure may its outcome be contested?"],
         g(["20.2", "25.4"], ["3 working days"], ["3 working days"], "A complaint is acknowledged within 3 working days."),
         g(["20.7", "25.6"], ["Employee complaint appeal", "5 working days of receiving the closure communication"], ["5 working days"],
           "A review may be requested within 5 working days of receiving the closure communication.")),
    base("sibling", ["How early must I cancel my cab for a hybrid office day if I will be working from home instead?",
                     "Hybrid day cab cancellation deadline?",
                     "Our team is moving to a hybrid schedule, so on some days I will work from home instead of coming in. How early before the pickup do I have to cancel the cab for those days?",
                     "What advance notice is needed to drop transport on a day I move from the workplace to remote work?"],
         g(["11.2", "11.4", "25.2"], ["6 hours", "6-hour"], ["6 hours", "6-hour"], "A hybrid-day allocation must be cancelled at least 6 hours before the journey.")),
    base("sibling", ["How many days can an ordinary recurring transport request cover in one submission?",
                     "Recurring booking length?",
                     "I travel the same route every working day, so I would rather not rebook each morning. What is the longest period I can set up in one go?",
                     "Over what span can a repeating transport arrangement be requested in a single submission?"],
         g(["4.3", "25.2"], ["Recurring booking horizon", "no more than 7 calendar days"], ["7 calendar days"],
           "An ordinary recurring request may cover no more than 7 calendar days.")),
    base("sibling", ["How late can I cancel my own scheduled trip without it counting as a no-show?",
                     "Ordinary trip cancellation deadline?",
                     "Something came up and I cannot take my usual cab home. How close to the departure time can I still cancel my own booking?",
                     "What is the latest point before departure at which an employee's scheduled journey may be called off?"],
         g(["4.2", "25.2"], ["Ordinary cancellation deadline", "up to 30 minutes before the scheduled time"], ["30 minutes"],
           "A trip can be cancelled up to 30 minutes before the scheduled time.")),
]

DEFINITIONS = {"acme": ACME, "globex": GLOBEX, "initech": INITECH}


# ------------------------------------------------------------------------------------------------ building

def section_body(markdown: str, section_id: str) -> str:
    m = re.search(rf"^### {re.escape(section_id)} .+?\n(.*?)(?=^### |^## |\Z)", markdown, flags=re.M | re.S)
    if not m:
        raise SystemExit(f"section {section_id} not found")
    return m.group(1).strip()


def golden_text(markdown: str, section_id: str, row_key=None) -> str:
    body = section_body(markdown, section_id)
    if row_key is None:
        return body
    table = [line for line in body.splitlines() if line.startswith("|")]
    rows = [line for line in table[2:] if row_key.lower() in line.lower()]
    if not rows:
        raise SystemExit(f"no row {row_key!r} in section {section_id}")
    return "\n".join([table[0], table[1], rows[0]])


def build(company: str) -> None:
    version = VERSIONS[company]
    markdown = (DATA / f"{company}_{version}_long.md").read_text(encoding="utf-8")
    facts, fact_index, lines = [], {}, []

    def fact_id(group) -> str:
        key = (tuple(group["sections"]), tuple(group["strings"]))
        if key not in fact_index:
            fact_index[key] = f"{company}-xf{len(fact_index) + 1:02d}"
            facts.append({"fact_id": fact_index[key], "summary": group["summary"], "section_ids": group["sections"],
                          "distinctive_strings": group["strings"], "statement_count": len(group["sections"])})
        return fact_index[key]

    for n, b in enumerate(DEFINITIONS[company], start=1):
        base_id = f"{company}-x{n:02d}"
        groups = b["groups"]
        sections = list(dict.fromkeys(s for grp in groups for s in grp["sections"]))
        golden, seen = [], set()
        for grp in groups:
            sid, row = grp["golden"] if grp["golden"] else (grp["sections"][0], None)
            if (sid, row) not in seen:
                seen.add((sid, row))
                golden.append({"section_id": sid, "text": golden_text(markdown, sid, row)})
        expected = {"status": "answered", "sections": sections}
        if len(groups) > 1:
            expected["required_sections"] = [grp["sections"] for grp in groups]
        expected["fact_ids"] = [[fact_id(grp)] for grp in groups]
        expected["must_contain_each"] = [grp["answer"] for grp in groups]
        expected["must_not_contain"] = []
        for variant, question in zip(VARIANTS, b["q"]):
            lines.append({"id": f"{base_id}-v{VARIANTS.index(variant)}", "base_id": base_id, "variant": variant,
                          "category": b["category"], "company": company, "profile": PROFILE[company], "question": question,
                          "expected": expected, "golden_answer": " ".join(grp["summary"] for grp in groups),
                          "golden_context": golden})

    stem = f"{company}_{version}"
    (DATA / f"{stem}_queries_extra.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in lines) + "\n", encoding="utf-8")
    (DATA / f"{stem}_facts_extra.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{company}: {len(DEFINITIONS[company])} base questions, {len(lines)} queries, {len(facts)} facts")


if __name__ == "__main__":
    for company in VERSIONS:
        build(company)
