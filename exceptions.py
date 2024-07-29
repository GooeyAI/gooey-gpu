import typing

import requests


class UserError(Exception):
    def __init__(
        self,
        message: str,
        sentry_level: str = "info",
        status_code: typing.Optional[int] = None,
    ):
        self.message = message
        self.sentry_level = sentry_level
        self.status_code = status_code
        super().__init__(
            dict(message=message, sentry_level=sentry_level, status_code=status_code)
        )

    def __str__(self):
        return self.message


def raise_for_status(resp: requests.Response, is_user_url: bool = False):
    """Raises :class:`HTTPError`, if one occurred."""

    http_error_msg = ""
    if isinstance(resp.reason, bytes):
        # We attempt to decode utf-8 first because some servers
        # choose to localize their reason strings. If the string
        # isn't utf-8, we fall back to iso-8859-1 for all other
        # encodings. (See PR #3538)
        try:
            reason = resp.reason.decode("utf-8")
        except UnicodeDecodeError:
            reason = resp.reason.decode("iso-8859-1")
    else:
        reason = resp.reason

    if 400 <= resp.status_code < 500:
        http_error_msg = f"{resp.status_code} Client Error: {reason} | URL: {resp.url} | Response: {_response_preview(resp)!r}"

    elif 500 <= resp.status_code < 600:
        http_error_msg = f"{resp.status_code} Server Error: {reason} | URL: {resp.url} | Response: {_response_preview(resp)!r}"

    if http_error_msg:
        exc = requests.HTTPError(http_error_msg, response=resp)
        if is_user_url:
            raise UserError(
                f"[{resp.status_code}] You have provided an invalid URL: {resp.url} "
                "Please make sure the URL is correct and accessible. ",
            ) from exc
        else:
            raise exc


def _response_preview(resp: requests.Response) -> bytes:
    return truncate_filename(resp.content, 500, sep=b"...")


def truncate_filename(
    text: typing.AnyStr, maxlen: int = 100, sep: typing.AnyStr = "..."
) -> typing.AnyStr:
    if len(text) <= maxlen:
        return text
    assert len(sep) <= maxlen
    mid = (maxlen - len(sep)) // 2
    return text[:mid] + sep + text[-mid:]
