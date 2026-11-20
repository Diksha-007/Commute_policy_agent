"""Mint a demo JWT (our stand-in for a real identity provider).

    python scripts/issue_token.py --company acme --grade L3 --shift night --gender female
    python scripts/issue_token.py --company acme --role admin

Prints only the token, so it can be captured:  TOKEN=$(python scripts/issue_token.py --company acme)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import issue_token  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--company", required=True, help="company_id, e.g. acme, globex, initech")
    p.add_argument("--employee", default="emp-001", help="employee id (sub claim)")
    p.add_argument("--role", default="employee", choices=["employee", "admin"])
    p.add_argument("--grade", help="e.g. L3")
    p.add_argument("--shift", help="e.g. night, general")
    p.add_argument("--gender", help="e.g. female, male")
    p.add_argument("--ttl", type=int, default=None, help="minutes until expiry (default from settings)")
    a = p.parse_args()
    print(issue_token(
        get_settings(), employee_id=a.employee, company_id=a.company, role=a.role,
        grade=a.grade, shift=a.shift, gender=a.gender, ttl_minutes=a.ttl,
    ))


if __name__ == "__main__":
    main()
