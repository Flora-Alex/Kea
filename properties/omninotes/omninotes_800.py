import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/done").click()
        

    @mainPath()
    def count_char_in_note_mainpath(self):
        d(resourceId="it.feio.android.omninotes:id/fab_expand_menu_button").long_click()
        d(resourceId="it.feio.android.omninotes:id/detail_content").set_text("Hello")
        d(resourceId="it.feio.android.omninotes:id/detail_title").set_text("Hello22")

    @precondition(lambda self: d(resourceId="it.feio.android.omninotes:id/menu_attachment").exists() and d(resourceId="it.feio.android.omninotes:id/menu_share").exists() and d(resourceId="it.feio.android.omninotes:id/menu_tag").exists()  )
    @rule()
    def count_char_in_note(self):
        title = d(resourceId="it.feio.android.omninotes:id/detail_title").get_text()
        print("title: " + str(title))
        content = d(resourceId="it.feio.android.omninotes:id/detail_content").get_text()
        print("content: " + str(content))
        if content == "Content":
            content = ""
        if title == "Title":
            title = ""
        import re
        number_of_char = len(re.findall(".",str(title))) + len(re.findall(".",str(content)))
        print("number of char: " + str(number_of_char))
        
        d(description="More options").click()
        
        d(text="Info").click()
        
        chars = int(d(resourceId="it.feio.android.omninotes:id/note_infos_chars").get_text())
        print("chars calculated by omninotes: " + str(chars))
        
        assert number_of_char == chars



if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/omninotes/OmniNotes-6.0.5.apk",
        device_serial="emulator-5554",
        output_dir="../output/omninotes/800/guided",
        policy_name="guided"
    )
    start_kea(t,setting)
    
