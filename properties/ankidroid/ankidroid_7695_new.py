import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        d(text="Get Started").click()

    @mainPath()
    def rename_note_type_shouldnot_display_mainpath(self):
        d(description="More options").click()
        d(text="Manage note types").click()
        d(text="Add note types").click()

    # 7695
    @precondition(
        lambda self: d(text="Manage note types").exists() and d(resourceId="com.ichi2.anki:id/note_name").exists()
    )
    @rule()
    def rename_note_type_shouldnot_display(self):
        d(resourceId="com.ichi2.anki:id/note_name").click()
        
        d(text="Front").click()
        
        d(text="Rename field").click()
        
        assert not d(text="Rename note type").exists(), "Rename note type found"




if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/ankidroid/2.18alpha6.apk",
        device_serial="emulator-5554",
        output_dir="output/ankidroid/7695/guided_new/1",
        policy_name="random"
    )
    start_kea(t,setting)
    
