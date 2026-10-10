# KYC Refresh Rules

*Northbridge Bank (fictional). Synthetic policy written for a demo project.*

## Purpose
Know Your Customer (KYC) information must stay current. These rules set how often it is refreshed and what blocks activity until it is.

## Risk tiers and refresh frequency
| Risk tier | Refresh every |
|-----------|---------------|
| standard  | 3 years |
| elevated  | 1 year |
| high      | 6 months |

## KYC statuses
- **verified**: identity and address confirmed and in date.
- **pending_review**: documents received but not yet checked, or a refresh is overdue.
- **unverified**: identity not yet confirmed.

## Restrictions by status
- **unverified** accounts cannot send international wires or add new payees.
- **pending_review** accounts may continue normal activity, but any transaction that triggers another fraud or AML alert must be escalated to a human reviewer rather than auto-cleared.
- **verified** accounts follow normal rules.

## Events that trigger an early refresh
- A sanctions screening hit, even if later cleared.
- A confirmed fraud case on the account.
- A sudden change in activity, such as volume many times above the monthly average.
- A change of country of residence.

## Business accounts
Business accounts must also confirm beneficial owners holding 25 percent or more.
