#my_prop.py
from kea.kea import *
from kea.kea_test import *

class CheckSearchBox(Kea):
    @precondition(True)
    @rule()
    def search_box_should_exist_after_rotation(self):
        d.rotate('l')
        d.rotate('n')
        assert d(resourceId="it.feio.android.omninotes.alpha:id/search_src_text").exists()