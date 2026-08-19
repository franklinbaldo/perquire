from perquire.providers.retry import (
    call_with_transient_retries,
    is_transient_provider_error,
    retry_after_seconds,
)


class RateLimitError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


def test_extracts_openrouter_retry_after_from_error_payload() -> None:
    error = RateLimitError('{"retry_after_seconds":12,"provider_error_code":"rate_limit_exceeded"}')
    assert retry_after_seconds(error) == 12.0


def test_retries_transient_failure_and_paces_every_attempt() -> None:
    attempts: list[int] = []
    paced: list[int] = []
    slept: list[float] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError('{"retry_after_seconds":2}')
        return "ok"

    result = call_with_transient_retries(
        operation,
        before_attempt=lambda: paced.append(1),
        max_retries=2,
        sleep=lambda seconds: slept.append(seconds),
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert len(paced) == 3
    assert slept == [2.0, 2.0]


def test_does_not_retry_non_transient_error() -> None:
    attempts: list[int] = []

    def operation() -> None:
        attempts.append(1)
        raise ValueError("bad request")

    try:
        call_with_transient_retries(operation, max_retries=3, sleep=lambda _: None)
    except ValueError:
        pass
    else:
        raise AssertionError("non-transient error should propagate")

    assert len(attempts) == 1


def test_transient_classification_covers_429_and_503() -> None:
    assert is_transient_provider_error(RateLimitError("429"))
    assert is_transient_provider_error(ServiceUnavailableError("503"))
    assert not is_transient_provider_error(ValueError("401 invalid credential"))
