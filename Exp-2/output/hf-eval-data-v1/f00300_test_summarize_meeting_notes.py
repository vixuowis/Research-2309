def test_summarize_meeting_notes():
    meeting_notes = "During the meeting, we discussed the progress of the project. We also talked about the challenges we are facing and the possible solutions. The team agreed to focus more on the quality of the work rather than the speed."
    summary = summarize_meeting_notes(meeting_notes)
    assert len(summary) < len(meeting_notes), "The summary is not shorter than the original text."
    assert isinstance(summary, str), "The output is not a string."

test_summarize_meeting_notes()