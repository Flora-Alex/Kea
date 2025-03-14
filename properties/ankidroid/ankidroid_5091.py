import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    
    @mainPath()
    def option_on_one_filter_deck_should_work_mainpath(self):
        d(description="More options").click()
        d(text="Create filtered deck").click()
        d(text="CREATE").click()

    @precondition(
        lambda self: 
        d(textContains="Options for Filtered Deck").exists()
    )
    @rule()
    def option_on_one_filter_deck_should_work(self):
        deck = d(textContains="Options for Filtered Deck").get_text()
        deck_name = str(deck.split("for ")[1])
        print("deck_name: " + deck_name)
        d(text="Search").click()
        d(className="android.widget.EditText").set_text("is:new")
        d(text="OK").click()
        d(text="Limit to").click()
        d(className="android.widget.EditText").set_text("1")
        d(text="OK").click()
        d(text="Reschedule").click()
        d(text="Reschedule").click()
        d.press("back")
        if d(resourceId="com.ichi2.anki:id/action_empty").exists() or d(text="STUDY").exists():
            d.press("back")

        assert d(text=deck_name).right(resourceId="com.ichi2.anki:id/deckpicker_new").get_text() == "1", "deck_name: " + deck_name + " count: " + d(text=deck_name).right(resourceId="com.ichi2.anki:id/deckpicker_new").get_text()



if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/ankidroid/2.9.7.apk",
        device_serial="emulator-5554",
        output_dir="../output/ankidroid/5091/guided",
        policy_name="guided"
    )
    start_kea(t,setting)
    
