# Put the code for your API here.
def test_answer(cmdopt = "type1"):
    if cmdopt == "type1":
        print("first")
    elif cmdopt == "type2":
        print("second")
    assert 1  # to see what was printed