# OpenRouter live viability ladder

This ladder deliberately increases live API cost only after the previous gate succeeds.

1. Credential smoke: authenticate `OPENROUTER_API_KEY` without running the experiment. Passed in Actions run 32249075367.
2. Provider boundary: #30 adds OpenRouter generation + embeddings, auditable accounting, parser integrity, and one shared upstream rate limiter across both provider roles.
3. One-case smoke: 1 hidden case, budget 2, three methods. This exercises the complete live path with 11 provider requests.
4. Mini benchmark: 3 hidden cases, budget 3, three methods. This is the first small empirical comparison rather than merely a connectivity test.
5. Full benchmark v1: 24 frozen cases at the selected budget. Run only after the smaller gates show the provider path is stable and the experiment produces interpretable traces.

At every gate, failure is evidence. Provider/model incompatibility, empty generations, invalid embeddings, budget-accounting drift, source leakage, rate-limit failures, or an adaptive method that fails to improve over simple baselines all block escalation rather than being hidden by retries or protocol changes.
