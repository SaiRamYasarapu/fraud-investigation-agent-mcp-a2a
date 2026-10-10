# Structuring Detection Thresholds

*Northbridge Bank (fictional). Synthetic policy written for a demo project.*

## Background
Cash transactions above $10,000 in a day require a Currency Transaction Report. Structuring means deliberately splitting cash into smaller amounts to avoid that report. Structuring is itself an offence, even if the money is legitimate.

## Alert rules
An alert is raised when:
- Three or more cash deposits between $8,000 and $9,999 are made within seven days, or
- Cash deposits made by different people into the same account total more than $10,000 in a day, or
- Money orders or cashier's checks under $3,000 are bought at several branches and deposited into one account.

## Reviewing an alert
1. Compare the pattern with the customer's history. A business that has deposited similar amounts on a steady schedule for years is a lower concern.
2. Check whether cash is quickly moved out, especially overseas. Rapid outbound movement raises the concern.
3. Look for several depositors acting together.

## Outcomes
- If the pattern is explained by normal, long-running business activity, document the reason and close.
- Otherwise escalate to AML compliance for a Suspicious Activity Report (SAR) decision.

## Never
Never advise a customer how to keep deposits below the reporting threshold.
