from src.preprocess.caption_events import split_events


def test_then_sequence():
    ev = split_events("A man runs, then he jumps, finally he lands.")
    assert ev == ["A man runs", "he jumps", "he lands"]


def test_followed_by():
    ev = split_events("A close-up of nail art, followed by a display of brushes.")
    assert ev == ["A close-up of nail art", "a display of brushes"]


def test_mid_before_keeps_order():
    # "A before B": A가 시간상 먼저
    ev = split_events("A girl hula hoops indoors before the scene shifts outdoors.")
    assert ev == ["A girl hula hoops indoors", "the scene shifts outdoors"]


def test_mid_after_inverts():
    # "A after B": B가 시간상 먼저
    ev = split_events("The man drinks water after he finishes the race.")
    assert ev == ["he finishes the race", "The man drinks water"]


def test_leading_after_keeps_order():
    ev = split_events("After the sun sets, the lights turn on.")
    assert ev == ["the sun sets", "the lights turn on"]


def test_leading_before_inverts():
    ev = split_events("Before the show starts, the crowd gathers.")
    assert ev == ["the crowd gathers", "the show starts"]


def test_as_while_not_split():
    # 동시 서술은 이벤트를 나누지 않는다
    ev = split_events("A skier moves forward as the camera zooms in.")
    assert len(ev) == 1


def test_real_train_caption():
    s = (
        "A girl hula hoops indoors before the scene shifts outdoors to a cheering "
        "group on rocks; then, players swim towards the pool's center, with one in "
        "a white cap preparing to pass the ball as spectators watch."
    )
    ev = split_events(s)
    assert len(ev) == 3
    assert ev[0].startswith("A girl hula hoops")
    assert ev[1].startswith("the scene shifts")
    assert ev[2].startswith("players swim")


def test_empty():
    assert split_events("") == []
    assert split_events(None) == []
