# Account Takeover Response Playbook

*Northbridge Bank (fictional). Synthetic policy written for a demo project.*

## What account takeover looks like
Account takeover (ATO) is when someone other than the customer gains control of their online access. Common warning signs:
- A password reset followed quickly by a new payee or a large transfer.
- Contact details (email, phone, address) changed shortly before money moves.
- Many failed logins, then a success from a new device or location.
- A mobile number recently ported to a new carrier (SIM swap).
- Logins through proxy or anonymizing services.
- A replacement card requested to a newly changed address.

## First response (within 15 minutes)
1. Freeze outgoing payments on the account.
2. End all active online sessions and revoke trusted devices.
3. Stop any pending wires or transfers that have not settled.

## Contact the customer
Use contact details on file from **before** any recent change. Confirm whether they made the recent changes and payments.

## If takeover is confirmed
- Reverse unauthorized contact changes.
- Reset credentials and move the customer to app-based or hardware authentication rather than SMS codes.
- Request recalls for any transfers that left the bank.
- File a fraud case and note the entry point if known (phishing, SIM swap, credential stuffing).

## If the customer made the changes themselves
Record the confirmation, release the restrictions, and note the pattern so the same activity does not trigger repeatedly.
