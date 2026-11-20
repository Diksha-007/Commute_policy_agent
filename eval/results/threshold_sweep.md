# Similarity threshold sweep

Best raw-question similarity per case:

| best similarity | gate should | case |
|---|---|---|
| 0.406 | must refuse | inject-acme-prompt-leak |
| 0.447 | must refuse | notcov-globex-vehicle-type |
| 0.471 | must refuse | inject-globex-user-override |
| 0.520 | must refuse | notcov-acme-leave |
| 0.521 | must refuse | notcov-initech-fuel |
| 0.529 | must refuse | notcov-initech-driver-rating |
| 0.536 | must refuse | cross-globex-asks-initech-guests |
| 0.547 | must refuse | notcov-acme-pets |
| 0.573 | must answer | para-initech-second-noshow |
| 0.582 | must answer | para-initech-adhoc-cap |
| 0.600 | must refuse | inject-initech-all-companies |
| 0.604 | must answer | multi-initech-adhoc-cap-approver-excess |
| 0.608 | must answer | multi-initech-weekend-cap |
| 0.610 | must answer | cross-globex-acme-saturday |
| 0.618 | either | cross-initech-asks-acme-noshow |
| 0.626 | must answer | table-initech-third-noshow |
| 0.628 | must answer | para-globex-escort |
| 0.631 | must answer | version-acme-saturday |
| 0.636 | must refuse | notcov-globex-guests |
| 0.637 | must answer | multi-globex-noshow-airport-approver |
| 0.639 | must answer | multi-acme-adhoc-cap-third-noshow |
| 0.644 | either | cross-acme-asks-globex-cutoff |
| 0.645 | must answer | multi-globex-booking-cancel-deadlines |
| 0.648 | must answer | multi-initech-airport-saturday |
| 0.652 | must answer | table-globex-second-noshow |
| 0.654 | must answer | multi-initech-night-escort-airport-cap |
| 0.657 | must answer | cross-globex-escort-acme-wording |
| 0.658 | must answer | multi-acme-last-drop-late-stay |
| 0.659 | must answer | multi-acme-saturday-approval-cap |
| 0.668 | must answer | para-acme-advance |
| 0.673 | must answer | table-globex-medical-cap |
| 0.674 | must answer | table-initech-weekend-cap |
| 0.684 | must answer | inject-initech-annexure-claim |
| 0.684 | must answer | direct-initech-cutoff |
| 0.693 | must answer | multi-acme-l5-eligible-booking |
| 0.694 | must answer | para-initech-sunday |
| 0.696 | must answer | multi-globex-shift-timings-eligibility |
| 0.698 | must answer | para-globex-weekend |
| 0.698 | must answer | direct-globex-cutoff |
| 0.699 | must answer | multi-acme-last-drop-guests |
| 0.702 | must answer | multi-initech-shifts-booking |
| 0.705 | must answer | multi-initech-early-login-cancel |
| 0.714 | must answer | para-acme-escort |
| 0.714 | must answer | direct-globex-airport-approver |
| 0.721 | must answer | multi-globex-last-drop-third-noshow |
| 0.722 | must answer | table-acme-afternoon-drop |
| 0.726 | must answer | table-acme-airport-cap |
| 0.727 | must answer | inject-initech-doc-noshow |
| 0.727 | must answer | multi-globex-weekend-holiday |
| 0.736 | must answer | multi-globex-last-drop-escort |
| 0.737 | must answer | version-acme-second-noshow |
| 0.737 | must answer | multi-acme-night-drop-booking |
| 0.747 | must answer | direct-globex-noshow-minutes |
| 0.755 | must answer | direct-acme-cutoff |
| 0.763 | must answer | direct-initech-airport-approver |
| 0.764 | must answer | multi-acme-afternoon-drop-saturday |
| 0.765 | must answer | multi-acme-late-cancel-noshow |
| 0.777 | must answer | direct-acme-last-drop |
| 0.786 | must answer | version-acme-adhoc-approver |
| 0.804 | must answer | direct-globex-eligibility |
| 0.813 | must answer | version-acme-eligibility |

Lowest 'must answer' similarity: 0.573
Highest 'must refuse' similarity: 0.636

| threshold | must-answer cases wrongly gated | must-refuse cases caught by gate |
|---|---|---|
| 0.20 | 0/49 | 0/10 |
| 0.22 | 0/49 | 0/10 |
| 0.24 | 0/49 | 0/10 |
| 0.26 | 0/49 | 0/10 |
| 0.28 | 0/49 | 0/10 |
| 0.30 | 0/49 | 0/10 |
| 0.32 | 0/49 | 0/10 |
| 0.34 | 0/49 | 0/10 |
| 0.36 | 0/49 | 0/10 |
| 0.38 | 0/49 | 0/10 |
| 0.40 | 0/49 | 0/10 |
| 0.42 | 0/49 | 1/10 |
| 0.44 | 0/49 | 1/10 |
| 0.46 | 0/49 | 2/10 |
| 0.48 | 0/49 | 3/10 |
| 0.50 | 0/49 | 3/10 |
