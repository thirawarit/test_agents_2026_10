# Platform Handbook

This handbook describes how the platform team runs production services.

## Backup Policy

Full backups of the customer database run every Sunday at 01:00 UTC and are retained for 90 days. Incremental backups run every six hours. All backup archives are copied to the Singapore object store, and the on-call engineer reviews the backup report every Monday.

### Restore Drills

The team runs a restore drill twice a year and must bring the customer database back within two hours. Every drill ends with a written review that lists what slowed the restore down.

## Deployment

Production releases go out on Tuesdays and Thursdays after the staging checks pass.

```bash
# Not a heading: this comment lives inside a code fence
./deploy.sh --env production --canary 10
```

A canary serves ten percent of traffic for thirty minutes before the full rollout.

## Incident Response

Severity is assigned by the first responder, who also opens the incident channel.

| Severity | Meaning | Response target |
| --- | --- | --- |
| SEV1 | Customer-facing outage | 15 minutes |
| SEV2 | Degraded service | 1 hour |
| SEV3 | Minor defect | Next business day |

## Office Plants

The ferns on the third floor are watered on Wednesdays by the facilities team.
