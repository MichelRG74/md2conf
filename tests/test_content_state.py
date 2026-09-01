"""
Publish Markdown files to Confluence wiki.

Copyright 2022-2026, Levente Hunyadi

:see: https://github.com/hunyadi/md2conf
"""

import json
import logging
import unittest
from argparse import ArgumentParser
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from cattrs.errors import BaseValidationError
from requests import HTTPError, Response, Session

from md2conf.api_types import ConfluenceContentState
from md2conf.api_v2 import ConfluenceSessionV2
from md2conf.clio import add_arguments, get_options
from md2conf.compatibility import override
from md2conf.environment import ConfluenceError, PageError
from md2conf.options import ProcessorOptions
from md2conf.options_api import ConfluenceSessionOptions
from md2conf.options_converter import ConverterOptions
from md2conf.publisher import Publisher
from tests.api import MockConfluenceAPI, MockConfluenceSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s [%(lineno)d] - %(message)s",
)


@contextmanager
def _create_temporary_directory() -> Generator[Path]:
    "Creates a temporary directory."

    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        yield Path(temp_dir)


def _get_processor_options(content_state: str | None) -> ProcessorOptions:
    return ProcessorOptions(
        content_state=content_state,
        converter=ConverterOptions(
            render_drawio=False,
            render_mermaid=False,
            render_plantuml=False,
            render_latex=False,
        ),
    )


class TestContentStateOption(unittest.TestCase):
    "Checks that `--content-state` is parsed like any other nullable command-line option."

    def test_parse_content_state(self) -> None:
        parser = ArgumentParser()
        add_arguments(parser, ProcessorOptions)

        args = parser.parse_args(["--content-state", "Verified"])
        options = get_options(args, ProcessorOptions)
        self.assertEqual(options.content_state, "Verified")

    def test_content_state_absent_by_default(self) -> None:
        parser = ArgumentParser()
        add_arguments(parser, ProcessorOptions)

        args = parser.parse_args([])
        options = get_options(args, ProcessorOptions)
        self.assertIsNone(options.content_state)

    def test_no_content_state_flag(self) -> None:
        parser = ArgumentParser()
        add_arguments(parser, ProcessorOptions)

        args = parser.parse_args(["--content-state", "Verified", "--no-content-state"])
        options = get_options(args, ProcessorOptions)
        self.assertIsNone(options.content_state)


class _NoContentStateCallsSession(MockConfluenceSession):
    """
    Fails the test if Content State *resolution by name* happens, used to verify --content-state has no
    such effect when omitted. Unlike resolution, preserving a state across a content republish (reading it
    via get_content_state and restoring it via set_content_state if one was already assigned) is unconditional
    and expected regardless of --content-state; it is exercised by other tests, not guarded here.
    """

    @override
    def get_available_content_states(self, page_id: str) -> list[ConfluenceContentState]:
        raise AssertionError("get_available_content_states should not be called when --content-state is omitted")


class _FailingUpdateSession(MockConfluenceSession):
    "Simulates a page update that fails, used to verify no Content State is assigned after a failed publish."

    @override
    def update_page(self, page_id: str, content: str, *, title: str, version: int, message: str) -> None:
        raise ConfluenceError("simulated page update failure")


class TestContentStateAssignment(unittest.TestCase):
    "Checks Content State resolution and assignment as part of the publication flow."

    def test_unchanged_behavior_when_option_omitted(self) -> None:
        "No Content State is resolved by name or assigned when `--content-state` is not supplied."

        with _create_temporary_directory() as source_dir:
            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nBody text.\n", encoding="utf-8")

            api = _NoContentStateCallsSession()
            try:
                Publisher(api, _get_processor_options(None)).process_page(document_path)
                page = api.get_page_properties_by_title("Document")
                self.assertIsNone(api.get_assigned_content_state(page.id))
            finally:
                api.close()

    def test_exact_name_resolution_and_assignment(self) -> None:
        "A page is assigned the Content State whose display name matches exactly."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            expected_id = api.add_space_content_state("Verified", color="Green")
            api.add_space_content_state("Ready for review", color="Yellow")

            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nBody text.\n", encoding="utf-8")

            Publisher(api, _get_processor_options("Verified")).process_page(document_path)

            page = api.get_page_properties_by_title("Document")
            self.assertEqual(api.get_assigned_content_state(page.id), expected_id)

    def test_assignment_after_page_creation(self) -> None:
        "A newly created page is assigned the requested Content State."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            expected_id = api.add_space_content_state("Verified")

            document_path = source_dir / "index.md"
            document_path.write_text("# New Document\n\nBody text.\n", encoding="utf-8")

            self.assertIsNone(api.page_exists("New Document"))
            Publisher(api, _get_processor_options("Verified")).process_page(document_path)

            page = api.get_page_properties_by_title("New Document")
            self.assertEqual(api.get_assigned_content_state(page.id), expected_id)

    def test_assignment_after_page_update(self) -> None:
        "An existing page, explicitly associated by page ID, is assigned the requested Content State on update."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            expected_id = api.add_space_content_state("Verified")

            homepage_id = api.get_homepage_id("SPACE_ID")
            existing_page = api.create_page(title="Existing Document", content="", parent_id=homepage_id, space_id="SPACE_ID")
            self.assertIsNone(api.get_assigned_content_state(existing_page.id))

            document_path = source_dir / "index.md"
            document_path.write_text(
                f"<!-- confluence-page-id: {existing_page.id} -->\n# Existing Document\n\nUpdated body text.\n",
                encoding="utf-8",
            )

            Publisher(api, _get_processor_options("Verified")).process_page(document_path)

            self.assertEqual(api.get_assigned_content_state(existing_page.id), expected_id)

    def test_no_assignment_if_publication_fails(self) -> None:
        "No Content State is assigned when the page content fails to publish."

        with _create_temporary_directory() as source_dir:
            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nBody text.\n", encoding="utf-8")

            api = _FailingUpdateSession()
            try:
                api.add_space_content_state("Verified")
                with self.assertRaises(ConfluenceError):
                    Publisher(api, _get_processor_options("Verified")).process_page(document_path)

                # the page is created by structure synchronization before content synchronization fails
                page_id = api.page_exists("Document")
                self.assertIsNotNone(page_id)
                if page_id is not None:
                    self.assertIsNone(api.get_assigned_content_state(page_id))
            finally:
                api.close()

    def test_requested_state_not_available(self) -> None:
        "Raises when no available Content State matches the requested name."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            api.add_space_content_state("Ready for review")

            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nBody text.\n", encoding="utf-8")

            with self.assertRaises(PageError):
                Publisher(api, _get_processor_options("Verified")).process_page(document_path)

    def test_duplicate_exact_name_matches(self) -> None:
        "Raises when more than one available Content State shares the exact requested name."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            api.add_space_content_state("Verified", color="Green")
            api.add_space_content_state("Verified", color="Blue")

            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nBody text.\n", encoding="utf-8")

            with self.assertRaises(PageError):
                Publisher(api, _get_processor_options("Verified")).process_page(document_path)

    def test_multiple_pages_assign_independently(self) -> None:
        "Each page in a multi-page publish resolves and assigns its own Content State independently."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            expected_id = api.add_space_content_state("Verified")

            (source_dir / "index.md").write_text("# Root Document\n\nBody text.\n", encoding="utf-8")
            (source_dir / "doc1.md").write_text("# First Document\n\nBody text.\n", encoding="utf-8")
            (source_dir / "doc2.md").write_text("# Second Document\n\nBody text.\n", encoding="utf-8")

            Publisher(api, _get_processor_options("Verified")).process_directory(source_dir)

            root_page = api.get_page_properties_by_title("Root Document")
            page1 = api.get_page_properties_by_title("First Document")
            page2 = api.get_page_properties_by_title("Second Document")

            self.assertNotEqual(root_page.id, page1.id)
            self.assertNotEqual(page1.id, page2.id)
            self.assertEqual(api.get_assigned_content_state(root_page.id), expected_id)
            self.assertEqual(api.get_assigned_content_state(page1.id), expected_id)
            self.assertEqual(api.get_assigned_content_state(page2.id), expected_id)

    def test_content_state_preserved_across_republish_without_option(self) -> None:
        "A Content State assigned by an earlier run survives a later content-only republish."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            verified_id = api.add_space_content_state("Verified")

            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nOriginal body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options("Verified")).process_page(document_path)

            page = api.get_page_properties_by_title("Document")
            self.assertEqual(api.get_assigned_content_state(page.id), verified_id)

            # republish with new content and no --content-state: MockConfluenceSession.update_page
            # simulates Confluence clearing the state, so this only passes if md2conf restores it.
            document_path.write_text("# Document\n\nUpdated body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options(None)).process_page(document_path)

            self.assertEqual(api.get_assigned_content_state(page.id), verified_id)

    def test_explicit_content_state_overrides_restored_state(self) -> None:
        "An explicit --content-state on a later run wins over whatever state would otherwise be restored."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            verified_id = api.add_space_content_state("Verified")
            ready_id = api.add_space_content_state("Ready for review")

            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nOriginal body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options("Verified")).process_page(document_path)

            page = api.get_page_properties_by_title("Document")
            self.assertEqual(api.get_assigned_content_state(page.id), verified_id)

            document_path.write_text("# Document\n\nUpdated body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options("Ready for review")).process_page(document_path)

            self.assertEqual(api.get_assigned_content_state(page.id), ready_id)

    def test_no_state_to_restore_is_a_no_op(self) -> None:
        "Republishing a page that never had a Content State assigned neither errors nor assigns one."

        with MockConfluenceAPI() as api, _create_temporary_directory() as source_dir:
            document_path = source_dir / "index.md"
            document_path.write_text("# Document\n\nOriginal body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options(None)).process_page(document_path)

            page = api.get_page_properties_by_title("Document")
            self.assertIsNone(api.get_assigned_content_state(page.id))

            document_path.write_text("# Document\n\nUpdated body text.\n", encoding="utf-8")
            Publisher(api, _get_processor_options(None)).process_page(document_path)

            self.assertIsNone(api.get_assigned_content_state(page.id))


def _make_session_v2() -> ConfluenceSessionV2:
    "Builds a REST API v2 session against a plain `requests.Session` with an explicit API URL to skip network probing."

    session = Session()
    return ConfluenceSessionV2(
        session,
        options=ConfluenceSessionOptions(),
        api_url="https://example.atlassian.net/wiki/",
        domain="example.atlassian.net",
        base_path="/wiki/",
        space_key="SPACE_KEY",
    )


def _json_response(status_code: int, payload: object) -> Response:
    response = Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")  # pyright: ignore[reportPrivateUsage]
    return response


class TestContentStateHttpShape(unittest.TestCase):
    """
    Checks the raw REST API v1 request/response shape used to query and assign Content States from a Confluence
    Cloud (REST API v2) publishing session, per <https://developer.atlassian.com/cloud/confluence/rest/v1/api-group-content-states/>.
    """

    def test_get_content_state_request_shape(self) -> None:
        api = _make_session_v2()
        api._session.get = Mock(  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
            return_value=_json_response(
                200,
                {"contentState": {"id": 73, "name": "Verified", "color": "Green"}, "lastUpdated": "2026-01-01T00:00:00.000Z"},
            )
        )

        state = api.get_content_state("123456")

        self.assertEqual(state, ConfluenceContentState(id=73, name="Verified", color="Green"))
        called_url = api._session.get.call_args.args[0]  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        # `status` is required, same as for `set_content_state`; extra fields like `lastUpdated`
        # in the response (not part of `ConfluenceContentStateResponse`) are ignored, not an error.
        self.assertEqual(called_url, "https://example.atlassian.net/wiki/rest/api/content/123456/state?status=current")

    def test_get_content_state_returns_none_when_unassigned(self) -> None:
        api = _make_session_v2()
        api._session.get = Mock(return_value=_json_response(404, {"message": "No content state found"}))  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

        self.assertIsNone(api.get_content_state("123456"))

    def test_get_content_state_request_failure(self) -> None:
        api = _make_session_v2()
        api._session.get = Mock(return_value=_json_response(500, {"message": "Internal error"}))  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

        # a genuine failure (not "no state assigned") must still propagate, not be swallowed as None
        with self.assertRaises(HTTPError):
            api.get_content_state("123456")

    def test_get_available_content_states_request_shape(self) -> None:
        api = _make_session_v2()
        api._session.get = Mock(  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
            return_value=_json_response(
                200,
                {
                    "spaceContentStates": [{"id": 73, "name": "Verified", "color": "Green"}],
                    "customContentStates": [{"id": 80, "name": "Draft", "color": "Grey"}],
                },
            )
        )

        states = api.get_available_content_states("123456")

        self.assertEqual(
            states,
            [
                ConfluenceContentState(id=73, name="Verified", color="Green"),
                ConfluenceContentState(id=80, name="Draft", color="Grey"),
            ],
        )
        api._session.get.assert_called_once()  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        called_url = api._session.get.call_args.args[0]  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        self.assertTrue(called_url.startswith("https://example.atlassian.net/wiki/rest/api/content/123456/state/available"))

    def test_set_content_state_request_shape(self) -> None:
        api = _make_session_v2()
        api._session.put = Mock(return_value=_json_response(200, {}))  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

        api.set_content_state("123456", 73)

        api._session.put.assert_called_once()  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        call = api._session.put.call_args  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        called_url = call.args[0]
        # `status` is required as a query parameter -- Confluence rejects the request with
        # "Invalid status 'null'" otherwise; it is not part of the JSON body.
        self.assertEqual(called_url, "https://example.atlassian.net/wiki/rest/api/content/123456/state?status=current")
        payload = json.loads(call.kwargs["data"])
        self.assertEqual(payload, {"id": 73})

    def test_get_available_content_states_malformed_response(self) -> None:
        api = _make_session_v2()
        # missing required fields `id` and `color` on the state object
        api._session.get = Mock(  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
            return_value=_json_response(200, {"spaceContentStates": [{"name": "Verified"}], "customContentStates": []})
        )

        with self.assertRaises(BaseValidationError):
            api.get_available_content_states("123456")

    def test_get_available_content_states_request_failure(self) -> None:
        api = _make_session_v2()
        api._session.get = Mock(return_value=_json_response(500, {"message": "Internal error"}))  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

        with self.assertRaises(HTTPError):
            api.get_available_content_states("123456")

    def test_set_content_state_request_failure(self) -> None:
        api = _make_session_v2()
        api._session.put = Mock(return_value=_json_response(400, {"message": "Bad request"}))  # type: ignore[method-assign] # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

        with self.assertRaises(HTTPError):
            api.set_content_state("123456", 999)


if __name__ == "__main__":
    unittest.main()
