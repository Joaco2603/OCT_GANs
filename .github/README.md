# GitHub Copilot Instructions - OCT_GANs

## Overview

This directory contains comprehensive instructions for AI coding assistants working on the OCT_GANs repository. These documents ensure consistent, high-quality contributions aligned with project standards.

## 📚 Documentation Structure

### Core Instructions

1. **[copilot-instructions.md](copilot-instructions.md)** - **START HERE**
   - Project overview and key locations
   - Model routing intelligence (which AI for which task)
   - Developer workflows and conventions
   - Quick reference for common tasks

### Specialized Guides

2. **[copilot-arquitecture.md](copilot-arquitecture.md)**
   - System architecture and design patterns
   - ProGAN model details (Generator, Discriminator)
   - Training pipeline and data flow
   - Performance optimization strategies

3. **[copilot-tests.md](copilot-tests.md)**
   - Testing philosophy and levels (smoke, unit, integration)
   - Test examples for model, data, checkpoints
   - GPU testing best practices
   - Coverage goals and common failures

4. **[copilot-documentation.md](copilot-documentation.md)**
   - Documentation standards and templates
   - README structure and style guide
   - Inline documentation (docstrings, comments)
   - Bilingual documentation (EN/ES)

5. **[copilot-contributing.md](copilot-contributing.md)**
   - Contribution workflow and branching strategy
   - PR process and review guidelines
   - Coding standards (PEP 8, type hints)
   - Recognition and community guidelines

6. **[copilot-guardrails.md](copilot-guardrails.md)**
   - Security policies and vulnerability reporting
   - Medical ethics and HIPAA/GDPR compliance
   - Intellectual property and licensing
   - AI assistant usage guidelines

## 🧠 Model Routing Intelligence

Before working on any task, determine the appropriate AI model:

```
┌─────────────────────────────────────────────────────────────┐
│                     Task Classification                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Architecture/Theory    → Claude Opus / GPT-4              │
│  Code Refactoring       → Claude Sonnet / GPT-4 Turbo      │
│  Quick Fixes/Docs       → GPT-3.5 / Claude Haiku           │
│  Data/Visualization     → Vision-capable model             │
│  Performance Tuning     → Claude Opus + profiling          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Example routing decisions**:
- *"Modify the generator architecture to add skip connections"* → **Claude Opus**
- *"Refactor data loader into separate utility module"* → **Claude Sonnet**
- *"Fix typo in README"* → **GPT-3.5**
- *"Analyze training loss curves from TensorBoard"* → **Vision model**
- *"Optimize batch size for memory efficiency"* → **Claude Opus + profiling**

## 🚀 Quick Start for AI Assistants

### First-Time Setup
1. Read `copilot-instructions.md` (15 min)
2. Skim `copilot-arquitecture.md` for system overview (10 min)
3. Review `copilot-tests.md` testing patterns (5 min)
4. Run smoke tests to verify understanding

### Before Each Task
1. **Classify task** using routing guide above
2. **Gather context**:
   - Read relevant code sections
   - Check related documentation
   - Review existing tests
3. **Plan approach**:
   - Identify files to modify
   - Consider testing strategy
   - Anticipate documentation updates

### After Completing Task
1. **Verify changes**:
   - Run relevant tests
   - Check code style (black, flake8)
   - Update documentation
2. **Self-review**:
   - Does it follow project conventions?
   - Are edge cases handled?
   - Is it well-documented?

## 📖 Usage Examples

### Example 1: Bug Fix

```
Task: "Fix CUDA memory leak when training at 512×512"

1. Routing: Performance issue → Claude Opus
2. Context: Read progan_local.py training loop
3. Investigation: Profile GPU memory usage
4. Fix: Add torch.cuda.empty_cache() after resolution transition
5. Test: Run memory test from copilot-tests.md
6. Document: Add inline comment explaining fix
```

### Example 2: New Feature

```
Task: "Add FID score evaluation during training"

1. Routing: Feature implementation → Claude Sonnet
2. Context: Read existing metric logging
3. Plan: 
   - Add fid_score.py in utils/
   - Integrate into training loop
   - Log to TensorBoard
4. Implement: Write code + tests
5. Document: Update API_REFERENCE.md and TRAINING_GUIDE.md
```

### Example 3: Documentation Update

```
Task: "Clarify batch size configuration for different GPUs"

1. Routing: Documentation → GPT-3.5
2. Context: Read current CONFIGURATION_GPU.md
3. Update: Add table with GPU models and recommended batch sizes
4. Translate: Update Spanish version (CONFIGURACION_GPU.md)
5. Verify: Test commands are accurate
```

## 🔍 Finding Information

### "Where do I find...?"

| Information Needed | Document | Section |
|-------------------|----------|---------|
| Project overview | copilot-instructions.md | Overview |
| Training workflow | copilot-instructions.md | Critical Developer Workflows |
| Model architecture | copilot-arquitecture.md | Core Components |
| Testing examples | copilot-tests.md | Test Levels |
| Documentation templates | copilot-documentation.md | Documentation Types |
| PR process | copilot-contributing.md | Pull Request Process |
| Security policies | copilot-guardrails.md | Security Policy |
| Medical ethics | copilot-guardrails.md | Data Privacy & Medical Ethics |

### "How do I...?"

| Task | Reference |
|------|-----------|
| Set up development environment | copilot-instructions.md → Environment Setup |
| Add a new model layer | copilot-arquitecture.md → Design Patterns |
| Write a unit test | copilot-tests.md → Unit Tests |
| Document a new function | copilot-documentation.md → Inline Documentation |
| Create a pull request | copilot-contributing.md → Development Workflow |
| Handle patient data | copilot-guardrails.md → Data Privacy & Medical Ethics |

## 🎯 Best Practices

### For AI Assistants

1. **Always start with routing** - Use the right model for the task
2. **Gather context first** - Don't make assumptions
3. **Follow conventions** - Match existing code style
4. **Test thoroughly** - Include tests with every change
5. **Document clearly** - Update docs alongside code
6. **Consider security** - Review guardrails for sensitive changes

### For Humans Working with AI

1. **Be specific** - "Fix GPU memory leak in training loop" vs "Fix memory"
2. **Provide context** - Share relevant file sections
3. **Verify outputs** - AI can make subtle mistakes
4. **Review guardrails** - Ensure AI follows security/ethics policies
5. **Give feedback** - Help AI improve with correction

## 🔄 Maintenance

### Updating These Instructions

When making changes:

1. **Propose update** via issue with label `documentation`
2. **Discuss rationale** with maintainers
3. **Update relevant files** (may span multiple guides)
4. **Update this README** if structure changes
5. **Notify contributors** of significant changes

### Keeping Instructions Current

- **Monthly review**: Check for outdated commands/paths
- **After major changes**: Update architecture diagrams
- **Before releases**: Full documentation audit

## 📊 Metrics

Track instruction effectiveness:

- Time to onboard new AI assistant: **Target < 30 min**
- Routing accuracy: **Target > 90%**
- Documentation coverage: **Target 100% of public APIs**
- Contribution compliance: **Target 95% follow guidelines**

## 🤝 Contributing to Instructions

See [copilot-contributing.md](copilot-contributing.md) for:
- How to propose improvements
- Documentation standards
- Review process

## 📞 Support

- **General questions**: Open issue with label `question`
- **Security concerns**: See copilot-guardrails.md → Security Policy
- **Technical discussions**: GitHub Discussions (if enabled)

---

## Document Status

| Document | Status | Last Updated | Next Review |
|----------|--------|--------------|-------------|
| copilot-instructions.md | ✅ Complete | 2025-11-12 | 2025-12-12 |
| copilot-arquitecture.md | ✅ Complete | 2025-11-12 | 2025-12-12 |
| copilot-tests.md | ✅ Complete | 2025-11-12 | 2025-12-12 |
| copilot-documentation.md | ✅ Complete | 2025-11-12 | 2025-12-12 |
| copilot-contributing.md | ✅ Complete | 2025-11-12 | 2025-12-12 |
| copilot-guardrails.md | ✅ Complete | 2025-11-12 | 2025-12-12 |

## Change Log

### 2025-11-12 - Initial Release
- Created comprehensive instruction set
- Added model routing intelligence
- Established documentation standards
- Defined security and ethics policies

---

**Maintained by**: OCT_GANs Team  
**License**: Same as project (see root LICENSE)  
**Feedback**: Open an issue or PR
