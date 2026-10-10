# Sanctions Screening and Escalation Procedure

*Northbridge Bank (fictional). Synthetic policy written for a demo project.*

## Purpose
To make sure the bank does not process payments for sanctioned people, companies, vessels or jurisdictions.

## What is screened
- Customer names at onboarding and whenever details change.
- Every party on outgoing and incoming payments: sender, beneficiary, beneficiary bank and intermediary banks.
- Trade documents for goods, vessels and ports.

## Lists used
OFAC SDN, UN Consolidated, EU Consolidated and UK HMT lists, refreshed daily.

## Match handling
### Exact match
Any exact match on name plus at least one other identifier (date of birth, nationality, address or registration number) is treated as a true match. The payment is blocked immediately and compliance is notified within one hour.

### Partial or name-only match
Common names often match by name alone. The analyst compares date of birth, nationality, address and other identifiers:
- If two or more identifiers clearly differ, record the match as a **false positive** with the evidence and release the payment.
- If identifiers are missing or unclear, keep the payment on hold and request more information from the customer.

### Spelling variants
Names may appear with different spellings or transliterations (for example, a single versus double letter). Screening uses fuzzy matching; a similarity score of 90 or above is treated as a possible match and reviewed.

## Ownership rule
A company owned 50 percent or more by sanctioned parties is treated as sanctioned even if it is not listed itself.

## Never
- Never tip off the customer that a sanctions review is in progress.
- Never split or reroute a payment to avoid a screening hit.
