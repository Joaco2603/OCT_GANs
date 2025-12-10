# Security & Compliance Guardrails - OCT_GANs

## Security Rules

### Vulnerability Reporting
- ❌ **DO NOT** open public issues for security vulnerabilities
- ✅ Email maintainers with description, reproduction steps, impact
- ✅ Wait for coordinated disclosure

### Secure Coding
- ❌ No arbitrary file paths from user input
- ✅ Whitelist `weights/` directory for checkpoints
- ✅ Validate file paths to prevent traversal
- ✅ Use `torch.load(..., weights_only=True)` for PyTorch 1.13+
- ✅ Enforce file size limits (max 5GB for checkpoints)
- ❌ Never hardcode credentials
- ✅ Use environment variables for secrets
- ✅ Add `.env` to `.gitignore`

### Dependencies
- ✅ Pin all versions in `requirements.txt`
- ✅ Run `safety check` and `pip-audit` periodically
- ❌ No unpinned or overly permissive versions

---

## Data Privacy & Medical Ethics

### PHI Compliance
- ✅ Use de-identified OCT2017 public dataset only
- ✅ Generate synthetic images (no real patient data)
- ❌ **NO** identifiable patient data processing

### Custom Data Requirements
- Obtain IRB approval
- De-identify: remove names, IDs, dates, DICOM metadata
- Encrypt at rest, restrict permissions, use SFTP

### Acceptable Use
- ✅ Research, education, data augmentation
- ❌ Clinical diagnosis, sharing PHI, unlicensed commercial use

### Medical Disclaimer
> ⚠️ **NOT a medical device**: Research only, NOT for clinical diagnosis or patient care

### Synthetic Data
- ✅ Label as "synthetic" in metadata
- ✅ Disclose generation method and limitations
- ❌ No misrepresentation as real data

---

## Intellectual Property

### Licensing
- Project license: **[INSERT LICENSE]**
- Contributions licensed under project license
- Contributors grant perpetual use/modify rights

### Third-Party Code
- Check license compatibility (MIT/BSD/Apache OK, GPL requires disclosure)
- Document attribution in code comments
- Add licenses to `LICENSES/` directory

### OCT2017 Dataset
- License: CC BY 4.0
- **Required citation**: Kermany et al., Mendeley Data V2, doi: 10.17632/rscbjbr9sj.2
- Verify rights for custom datasets

---

## Dependency Management

### Approved Core Dependencies
- PyTorch, torchvision, NumPy, Pillow

### Adding Dependencies
1. Open issue proposing dependency
2. Verify: necessity, license, maintenance status, security
3. Await approval
4. Pin version in `requirements.txt`
5. Test on clean environment

---

## Code Review Checklist

### File I/O
- [ ] Paths validated (no traversal)
- [ ] File size limits enforced
- [ ] Read-only permissions when possible

### Network Access
- [ ] HTTPS only
- [ ] Timeouts configured
- [ ] Credentials via environment variables

### User Input
- [ ] Validated and sanitized
- [ ] Whitelisted values
- [ ] No sensitive info in error messages

### Secrets
- [ ] No hardcoded passwords/keys
- [ ] `.env` in `.gitignore`

---

## AI Assistant Rules

### Allowed
- ✅ Code generation, refactoring, documentation, testing, bug fixes

### Not Allowed
- ❌ Violating licenses
- ❌ Using proprietary training data
- ❌ Bypassing security checks
- ❌ Misleading documentation

### AI-Generated Code
- Review thoroughly
- Verify no license violations
- Test extensively
- Optional: note AI assistance in commits

---

## Incident Response

### Security Incident
1. Contain (offline systems)
2. Assess scope/impact
3. Notify maintainers
4. Remediate and patch
5. Document postmortem

### Data Breach
1. Immediate containment
2. Legal notification (HIPAA: 60 days)
3. Forensic investigation
4. Additional safeguards

---

**Last Updated**: November 12, 2025
