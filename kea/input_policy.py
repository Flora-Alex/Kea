import os
import logging
import random
import copy
import time

from .utils import Time, generate_report, save_log, RULE_STATE
from abc import abstractmethod
from .input_event import (
    KEY_RotateDeviceToPortraitEvent,
    KEY_RotateDeviceToLandscapeEvent,
    KeyEvent,
    IntentEvent,
    ReInstallAppEvent,
    RotateDevice,
    RotateDeviceToPortraitEvent,
    RotateDeviceToLandscapeEvent,
    KillAppEvent,
    KillAndRestartAppEvent,
    SetTextEvent,
)
from .utg import UTG

# from .kea import utils
from .kea import CHECK_RESULT
from typing import TYPE_CHECKING, Dict

from .api_interface import PPOClient

if TYPE_CHECKING:
    from .input_manager import InputManager
    from .kea import Kea
    from .app import App
    from .device import Device

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 10
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5
START_TO_GENERATE_EVENT_IN_POLICY = 2
# Max number of query llm
MAX_NUM_QUERY_LLM = 10

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_GUIDED = "guided"
POLICY_RANDOM = "random"
POLICY_NONE = "none"
POLICY_LLM = "llm"


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device: "Device", app: "App", allow_to_generate_utg=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_recoder = Time()
        self.utg = UTG(device=device, app=app)
        self.device = device
        self.app = app
        self.event_count = 0

        self.last_event = None
        self.from_state = None
        self.to_state = None
        self.allow_to_generate_utg = allow_to_generate_utg
        self.triggered_bug_information = []
        self.time_needed_to_satisfy_precondition = []
        self.statistics_of_rules = {}

        self._num_restarts = 0
        self._num_steps_outside = 0
        self._event_trace = ""

    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        # number of events that have been executed
        self.event_count = 0
        # self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                # always try to close the keyboard on the device.
                # if self.device.is_harmonyos is False and hasattr(self.device, "u2"):
                #     self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration event count: %d", self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()

                # set the from_state to droidbot to let the pdl get the state
                self.device.from_state = self.from_state

                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    # start the application
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    event = self.generate_event()

                if event is not None:
                    try:
                        self.device.save_screenshot_for_report(
                            event=event, current_state=self.from_state
                        )
                    except Exception as e:
                        self.logger.error("SaveScreenshotForReport failed: %s", e)
                        self.from_state = self.device.get_current_state()
                        self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
                    input_manager.add_event(event)
                self.to_state = self.device.get_current_state()
                self.last_event = event





                if self.allow_to_generate_utg:
                    self.update_utg()

                bug_report_path = os.path.join(self.device.output_dir, "all_states")
                # TODO this function signature is too long?
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                    self.statistics_of_rules
                )

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback

                traceback.print_exc()
            self.event_count += 1
        self.tear_down()

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)

    def move_the_app_to_foreground_if_needed(self, current_state):
        """
        if the app is not running on the foreground of the device, then try to bring it back
        """
        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self._event_trace.endswith(
                    EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP
            ) or self._event_trace.endswith(EVENT_FLAG_START_APP):
                self._num_restarts += 1
                self.logger.info(
                    "The app had been restarted %d times.", self._num_restarts
                )
            else:
                self._num_restarts = 0

            # pass (START) through
            if not self._event_trace.endswith(EVENT_FLAG_START_APP):
                if self._num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                else:
                    # Start the app
                    self._event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self._num_steps_outside += 1

            if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self._event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self._num_steps_outside = 0

    @abstractmethod
    def tear_down(self):
        """ """
        pass

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    @abstractmethod
    def generate_random_event_based_on_current_state(self):
        """
        generate an event
        @return:
        """
        pass


class KeaInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, kea: "Kea" = None, allow_to_generate_utg=False):
        super(KeaInputPolicy, self).__init__(device, app, allow_to_generate_utg)
        self.kea = kea
        # self.last_event = None
        # self.from_state = None
        # self.to_state = None

        # retrive all the rules from the provided properties
        for rule in self.kea.all_rules:
            self.statistics_of_rules[str(rule.function.__name__)] = {
                RULE_STATE.PRECONDITION_SATISFIED: 0,
                RULE_STATE.PROPERTY_CHECKED: 0,
                RULE_STATE.POSTCONDITION_VIOLATED: 0,
                RULE_STATE.UI_OBJECT_NOT_FOUND: 0
            }

    def run_initializer(self):
        if self.kea.initializer is None:
            self.logger.warning("No initializer")
            return

        result = self.kea.execute_initializer(self.kea.initializer)
        if (
                result == CHECK_RESULT.PASS
        ):  # why only check `result`, `result` could have different values.
            self.logger.info("-------initialize successfully-----------")
        else:
            self.logger.error("-------initialize failed-----------")

    def check_rule_whose_precondition_are_satisfied(self):
        """
        TODO should split the function
        #! xixian - agree to split the function
        """
        # ! TODO - xixian - should we emphasize the following data structure is a dict?
        rules_ready_to_be_checked = (
            self.kea.get_rules_whose_preconditions_are_satisfied()
        )
        rules_ready_to_be_checked.update(self.kea.get_rules_without_preconditions())
        if len(rules_ready_to_be_checked) == 0:
            self.logger.debug("No rules match the precondition")
            return

        candidate_rules_list = list(rules_ready_to_be_checked.keys())
        # randomly select a rule to check
        rule_to_check = random.choice(candidate_rules_list)

        if rule_to_check is not None:
            self.logger.info(f"-------Check Property : {rule_to_check}------")
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PROPERTY_CHECKED
            ] += 1
            precondition_page_index = self.device.cur_event_count
            # check rule, record relavant info and output log
            result = self.kea.execute_rule(
                rule=rule_to_check, keaTest=rules_ready_to_be_checked[rule_to_check]
            )
            if result == CHECK_RESULT.ASSERTION_FAILURE:
                self.logger.error(
                    f"-------Postcondition failed. Assertion error, Property:{rule_to_check}------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.POSTCONDITION_VIOLATED
                ] += 1
                postcondition_page__index = self.device.cur_event_count
                self.triggered_bug_information.append(
                    (
                        (precondition_page_index, postcondition_page__index),
                        self.time_recoder.get_time_duration(),
                        rule_to_check.function.__name__,
                    )
                )
            elif result == CHECK_RESULT.PASS:
                self.logger.info(
                    f"-------Post condition satisfied. Property:{rule_to_check} pass------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )

            elif result == CHECK_RESULT.UI_NOT_FOUND:
                self.logger.error(
                    f"-------Execution failed: UiObjectNotFound during exectution. Property:{rule_to_check}-----------"
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.UI_OBJECT_NOT_FOUND
                ] += 1
            elif result == CHECK_RESULT.PRECON_NOT_SATISFIED:
                self.logger.info("-------Precondition not satisfied-----------")
            else:
                raise AttributeError(f"Invalid property checking result {result}")

    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)


class RandomPolicy(KeaInputPolicy):
    """
    generate random event based on current app state
    """

    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_reinstall_app=False,
            allow_to_generate_utg=False,
            disable_rotate=False,
            output_dir=None
    ):
        super(RandomPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.restart_app_after_check_property = restart_app_after_check_property
        self.number_of_events_that_restart_app = number_of_events_that_restart_app
        self.clear_and_reinstall_app = clear_and_reinstall_app
        self.logger = logging.getLogger(self.__class__.__name__)
        if output_dir is None:
            output_dir = "output"
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.disable_rotate = disable_rotate
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def generate_event(self):
        """
        generate an event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        if self.event_count % self.number_of_events_that_restart_app == 0:
            if self.clear_and_reinstall_app:
                self.logger.info(
                    "clear and reinstall app after %s events"
                    % self.number_of_events_that_restart_app
                )
                return ReInstallAppEvent(self.app)
            self.logger.info(
                "restart app after %s events" % self.number_of_events_that_restart_app
            )
            return KillAndRestartAppEvent(app=self.app)

        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()

        return event

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.debug("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        possible_events = current_state.get_possible_input()
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            # select a rotate event with different direction than last time
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = (
                    RotateDeviceToLandscapeEvent()
                )
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()
        return event


class GuidedPolicy(KeaInputPolicy):
    """
    generate events around the main path
    """

    def __init__(self, device, app, kea=None, allow_to_generate_utg=False, disable_rotate=False, output_dir=None):
        super(GuidedPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.disable_rotate = disable_rotate
        if len(self.kea.all_mainPaths):
            self.logger.info("Found %d mainPaths" % len(self.kea.all_mainPaths))
        else:
            self.logger.error("No mainPath found")

        self.main_path = None
        self.execute_main_path = True

        self.current_index_on_main_path = 0
        self.max_number_of_mutate_steps_on_single_node = 20
        self.current_number_of_mutate_steps_on_single_node = 0
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.index_on_main_path_after_mutation = -1
        self.mutate_node_index_on_main_path = 0

        self.last_random_text = None
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def select_main_path(self):
        if len(self.kea.all_mainPaths) == 0:
            self.logger.error("No mainPath")
            return
        self.main_path = random.choice(self.kea.all_mainPaths)
        # self.path_func, self.main_path =  self.kea.parse_mainPath(self.main_path)
        self.path_func, self.main_path = self.main_path.function, self.main_path.path
        self.logger.info(
            f"Select the {len(self.main_path)} steps mainPath function: {self.path_func}"
        )
        self.main_path_list = copy.deepcopy(self.main_path)
        self.max_number_of_events_that_try_to_find_event_on_main_path = min(
            10, len(self.main_path)
        )
        self.mutate_node_index_on_main_path = len(self.main_path)

    def generate_event(self):
        """ """
        current_state = self.from_state

        # Return relevant events based on whether the application is in the foreground.
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        if ((self.event_count == START_TO_GENERATE_EVENT_IN_POLICY)
                or isinstance(self.last_event, ReInstallAppEvent)):
            self.select_main_path()
            self.run_initializer()
            time.sleep(2)
            self.from_state = self.device.get_current_state()
        if self.execute_main_path:
            event_str = self.get_next_event_from_main_path()
            if event_str:
                self.logger.info("*****main path running*****")
                self.kea.execute_event_from_main_path(event_str)
                return None
        if event is None:
            # generate event aroud the state on the main path
            event = self.mutate_the_main_path()

        return event

    def stop_mutation(self):
        self.index_on_main_path_after_mutation = -1
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.execute_main_path = True
        self.current_number_of_mutate_steps_on_single_node = 0
        self.current_index_on_main_path = 0
        self.mutate_node_index_on_main_path -= 1
        if self.mutate_node_index_on_main_path == -1:
            self.mutate_node_index_on_main_path = len(self.main_path)
            return ReInstallAppEvent(app=self.app)
        self.logger.info(
            "reach the max number of mutate steps on single node, restart the app"
        )
        return KillAndRestartAppEvent(app=self.app)

    def mutate_the_main_path(self):
        event = None
        self.current_number_of_mutate_steps_on_single_node += 1

        if (
                self.current_number_of_mutate_steps_on_single_node
                >= self.max_number_of_mutate_steps_on_single_node
        ):
            # try to find an event from the main path that can be executed on current state
            if (
                    self.number_of_events_that_try_to_find_event_on_main_path
                    <= self.max_number_of_events_that_try_to_find_event_on_main_path
            ):
                self.number_of_events_that_try_to_find_event_on_main_path += 1
                # if reach the state that satsfies the precondition, check the rule and turn to execute the main path.
                if self.index_on_main_path_after_mutation == len(self.main_path_list):
                    self.logger.info(
                        "reach the end of the main path that could satisfy the precondition"
                    )
                    rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
                    for rule_to_check in rules_to_check:
                        self.statistics_of_rules[str(rule_to_check.function.__name__)][
                            RULE_STATE.PRECONDITION_SATISFIED
                        ] += 1
                    if len(rules_to_check) > 0:
                        t = self.time_recoder.get_time_duration()
                        self.time_needed_to_satisfy_precondition.append(t)
                        self.logger.debug(
                            "has rule that matches the precondition and the time duration is "
                            + t
                        )
                        self.logger.info("Check property")
                        self.check_rule_whose_precondition_are_satisfied()
                    return self.stop_mutation()

                # find if there is any event in the main path that could be executed on currenty state
                event_str = self.get_event_from_main_path()
                try:
                    self.kea.execute_event_from_main_path(event_str)
                    self.logger.info("find the event in the main path")
                    return None
                except Exception:
                    self.logger.info("can't find the event in the main path")
                    return self.stop_mutation()

            return self.stop_mutation()

        self.index_on_main_path_after_mutation = -1

        if len(self.kea.get_rules_whose_preconditions_are_satisfied()) > 0:
            # if the property has been checked, don't return any event
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()
        return event

    def get_next_event_from_main_path(self):
        """
        get a next event when execute on the main path
        """
        if self.current_index_on_main_path == self.mutate_node_index_on_main_path:
            self.logger.info(
                "reach the mutate index, start mutate on the node %d"
                % self.mutate_node_index_on_main_path
            )
            self.execute_main_path = False
            return None

        self.logger.info(
            "execute node index on main path: %d" % self.current_index_on_main_path
        )
        u2_event_str = self.main_path_list[self.current_index_on_main_path]
        if u2_event_str is None:
            self.logger.warning(
                "event is None on main path node %d" % self.current_index_on_main_path
            )
            self.current_index_on_main_path += 1
            return self.get_next_event_from_main_path()
        self.current_index_on_main_path += 1
        return u2_event_str

    def get_ui_element_dict(self, ui_element_str: str) -> Dict[str, str]:
        """
        get ui elements of the event
        """
        start_index = ui_element_str.find("(") + 1
        end_index = ui_element_str.find(")", start_index)

        if start_index != -1 and end_index != -1:
            ui_element_str = ui_element_str[start_index:end_index]
        ui_elements = ui_element_str.split(",")

        ui_elements_dict = {}
        for ui_element in ui_elements:
            attribute_name, attribute_value = ui_element.split("=")
            attribute_name = attribute_name.strip()
            attribute_value = attribute_value.strip()
            attribute_value = attribute_value.strip('"')
            ui_elements_dict[attribute_name] = attribute_value
        return ui_elements_dict

    def get_event_from_main_path(self):
        """
        get an event can lead current state to go back to the main path
        """
        if self.index_on_main_path_after_mutation == -1:
            for i in reversed(range(len(self.main_path_list))):
                event_str = self.main_path_list[i]
                ui_elements_dict = self.get_ui_element_dict(event_str)
                current_state = self.from_state
                view = current_state.get_view_by_attribute(ui_elements_dict)
                if view is None:
                    continue
                self.index_on_main_path_after_mutation = i + 1
                return event_str
        else:
            event_str = self.main_path_list[self.index_on_main_path_after_mutation]
            ui_elements_dict = self.get_ui_element_dict(event_str)
            current_state = self.from_state
            view = current_state.get_view_by_attribute(ui_elements_dict)
            if view is None:
                return None
            self.index_on_main_path_after_mutation += 1
            return event_str
        return None

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG to explore the app
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.info("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        # Get all possible input events
        possible_events = current_state.get_possible_input()

        # if self.random_input:
        #     random.shuffle(possible_events)
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event


class LLMPolicy(RandomPolicy):
    """
    use LLM to generate input when detected ui tarpit
    """

    def tear_down(self):
        pass

    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            output_dir=None,
    ):
        super(LLMPolicy, self).__init__(device, app, kea)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.__action_history = []
        self.store = {}
        self.__all_action_history = set()
        self.__activity_history = set()

        self.effective_event_strs = set()
        self.ineffective_event_strs = set()
        self.explored_state_strs = set()
        self.reached_activities = set()
        self.last_rules_whose_preconditions_are_satisfied = None
        self.cur_rules_whose_preconditions_are_satisfied = self.kea.get_rules_whose_preconditions_are_satisfied()

        self.from_state = None
        self.task = ("I am an expert in App GUI testing to guide the testing tool to enhance the coverage of "
                     "functional scenarios in testing the App based on my extensive App testing experience.")

        # if not os.environ.get("OPENAI_API_KEY"):
        #     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        self.api = PPOClient()
        self.api.attach()
        self.logger.info("API attached.")
        # self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        # number of events that have been executed
        self.event_count = 0
        # self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                # always try to close the keyboard on the device.
                # if self.device.is_harmonyos is False and hasattr(self.device, "u2"):
                #     self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration event count: %d", self.event_count)
                is_llm_event=False

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()

                # set the from_state to droidbot to let the pdl get the state
                self.device.from_state = self.from_state
                event = self.move_the_app_to_foreground_if_needed(self.from_state)
                if event is not None:
                    # Do nothing
                    pass
                elif self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    # start the application
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    event = self.generate_event()
                    is_llm_event = True

                if event is not None:
                    try:
                        self.device.save_screenshot_for_report(
                            event=event, current_state=self.from_state
                        )
                    except Exception as e:
                        self.logger.error("SaveScreenshotForReport failed: %s", e)
                        self.from_state = self.device.get_current_state()
                        self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
                    input_manager.add_event(event)
                self.to_state = self.device.get_current_state()
                self.last_event = event

                event_str = event.get_event_str(self.from_state)

                if self.from_state.state_str == self.to_state.state_str:
                    self.ineffective_event_strs.add(event_str)
                    if event_str in self.effective_event_strs:
                        self.effective_event_strs.remove(event_str)
                self.effective_event_strs.add(event_str)
                
                if is_llm_event:
                    reward=self.CalculateReward(event)
                    self.api.feedback(reward=reward,done=False)



                if self.allow_to_generate_utg:
                    self.update_utg()

                bug_report_path = os.path.join(self.device.output_dir, "all_states")
                # TODO this function signature is too long?
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                    self.statistics_of_rules
                )

            except KeyboardInterrupt:
                self.api.detach()
                self.logger.info("API detached.")
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback

                traceback.print_exc()
            self.event_count += 1
        self.tear_down()

    def CalculateReward(self, event, base_reward = 50):
        find_bug_rewards = {
            "FAILURE": 200,
            "PASS": 10,
            "UI_NOT_FOUND": 0,
            "PRECON_NOT_SATISFIED": 0,
            "NO_RULES": 0
        }
        find_new_event_reward = 10.0
        find_new_state_reward = 30.0
        find_new_rules_whose_preconditions_are_satisfied = 20.0
        error_punishment = 20.0
        # 更新rules_whose_preconditions_are_satisfied
        self.last_rules_whose_preconditions_are_satisfied = self.cur_rules_whose_preconditions_are_satisfied
        self.cur_rules_whose_preconditions_are_satisfied = self.kea.get_rules_whose_preconditions_are_satisfied()
        reward = base_reward
        # 找到bug
        # reward += find_bug_rewards[self.check_rule()]
        if len(self.cur_rules_whose_preconditions_are_satisfied) > len(
                self.last_rules_whose_preconditions_are_satisfied):
            reward += find_new_rules_whose_preconditions_are_satisfied
        # 探索到新event
        if not self.is_event_explored(event, self.from_state):
            reward += find_new_event_reward
        # 探索到新state
        if not self.is_state_explored(self.to_state):
            reward += find_new_state_reward
        # app不在前台
        if self.to_state.get_app_activity_depth(self.app) != 0:
            reward = error_punishment
        return reward



    def check_rule(self):

        rules_ready_to_be_checked = (
            self.kea.get_rules_whose_preconditions_are_satisfied()
        )
        rules_ready_to_be_checked.update(self.kea.get_rules_without_preconditions())
        if len(rules_ready_to_be_checked) == 0:
            return "NO_RULES"
        candidate_rules_list = list(rules_ready_to_be_checked.keys())
        # randomly select a rule to check
        rule_to_check = random.choice(candidate_rules_list)
        if rule_to_check is not None:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PROPERTY_CHECKED
            ] += 1
            precondition_page_index = self.device.cur_event_count
            # check rule, record relavant info and output log
            result = self.kea.execute_rule(
                rule=rule_to_check, keaTest=rules_ready_to_be_checked[rule_to_check]
            )

            if result == CHECK_RESULT.ASSERTION_FAILURE:
                return "FAILURE"
            elif result == CHECK_RESULT.PASS:
                return "PASS"
            elif result == CHECK_RESULT.UI_NOT_FOUND:
                return "UI_NOT_FOUND"
            elif result == CHECK_RESULT.PRECON_NOT_SATISFIED:
                return "PRECON_NOT_SATISFIED"
            else:
                raise AttributeError(f"Invalid property checking result {result}")

    def is_event_explored(self, event, state):

        event_str = event.get_event_str(state)

        return (

                event_str in self.effective_event_strs

                or event_str in self.ineffective_event_strs
        )

    def is_state_explored(self, state):
        if state.state_str in self.explored_state_strs:
            return True
        # PS: No need to check events here.
        self.explored_state_strs.add(state.state_str)
        return False





    # Generate a response using the retrieved content.
    def _get_action_with_LLM(self, current_state, action_history, activity_history):
        """
        Generate answer.
        """
        """
                markdown_path = "https://raw.githubusercontent.com/openatx/uiautomator2/master/README_CN.md"
                loader = UnstructuredMarkdownLoader(markdown_path)
                data = loader.load()
                assert len(data) == 1
                assert isinstance(data[0], Document)
                readme_content = data[0].page_content
                api_start_index = readme_content.find("API Documents")
                if api_start_index != -1:
                    api_content = readme_content[api_start_index:]
                else:
                    api_content = ""
                    print("没有找到API Documents")
                api_document = Document(page_content=api_content)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                all_splits = text_splitter.split_documents([api_document])

                self.vector_store = Chroma.from_documents(documents=all_splits, embedding=self.embeddings)
        docs = [Document(page_content="ui testing")]
        self.vector_store = Chroma.from_documents(documents=docs, embedding=self.embeddings)
        retriever = self.vector_store.as_retriever()
        """

        activity = current_state.foreground_activity
        task_prompt = (
                self.task
                + f"My task is to select an action based on the current GUI Infomation to perform next and help the app prevent entering or escape the UI tarpit."
        )

        visited_page_prompt = (
                f"Currently, the App is running on the {activity} page."
                f"I have already visited the following activities: \n"
                + "\n".join(activity_history)
        )

        for i in range(len(action_history)):
            if action_history[i] is None:
                action_history[i] = "- None"
        history_prompt = (
                f"I have already completed the following steps: \n "
                + ";\n ".join(action_history if len(action_history) > 0 else ["None"])
        )

        state_prompt, _ = current_state.get_described_actions()
        candidate_actions = current_state.get_possible_input()
        candidate_actions.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            candidate_actions.append(RotateDevice())

        actions = [f"{i}: {action.get_event_str(current_state)}" for i, action in enumerate(candidate_actions)]
        actions_prompt = (
                f"Here are the actions I can take: \n"
                + "\n".join(actions)
        )

        question = "Which action should I choose next? I shall choose No. "
        system_message_content = f"{task_prompt}\n{visited_page_prompt}\n{history_prompt}\n{state_prompt}\n{actions_prompt}\n{question}"

        obs = {"prompt": system_message_content, "action": actions}
        # print(system_message_content)

        # TODO: Adapt this to api interface
        status = self.api.ask()
        if status["status"] != "ok":
            # Maybe training, hold for 1s
            self.logger.error(status["status"])
            time.sleep(1)
            status = self.api.ask()
        actions_sampled = self.api.step(obs)
        selected_action = candidate_actions[actions_sampled["action"]]

        return selected_action, candidate_actions

    def generate_event(self):
        """
        generate an LLM event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        if (
                self.event_count % self.number_of_events_that_restart_app == 0
                and self.clear_and_reinstall_app
        ):
            self.logger.info(
                "clear and restart app after %s events"
                % self.number_of_events_that_restart_app
            )
            return ReInstallAppEvent(self.app)
        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info(
                    "Found exectuable property in current state. No property will be checked now according to the random checking policy."
                )

        event = self.generate_llm_event_based_on_utg()
        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event

    def generate_llm_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.debug("Current state: %s" % current_state.state_str)
        self._event_trace += EVENT_FLAG_EXPLORE

        action, candidate_actions = self._get_action_with_LLM(
            current_state,
            self.__action_history,
            self.__activity_history,
        )
        if action is not None:
            self.__action_history.append(current_state.get_action_desc(action))
            self.__all_action_history.add(current_state.get_action_desc(action))
            return action

    def get_last_state(self):
        return self.from_state

    def clear_action_history(self):
        self.__action_history = []
