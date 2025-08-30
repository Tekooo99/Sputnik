# -*- coding: utf-8 -*-

class SpecialTokens:
    def __init__(self):
        self.categories = {
            "uav": "<type:uav>",
            "airliner": "<type:airliner>",
            "cargo_aircraft": "<type:cargo-aircraft>",
            "spacecraft": "<type:spacecraft>",
            "general_aviation": "<type:general-aviation>",
            "business_jet": "<type:business-jet>",
            "military_fighter": "<type:military-fighter>",
            "military_bomber": "<type:military-bomber>",
            "military_transport": "<type:military-transport>",
            "helicopter": "<type:helicopter>",
            "historical": "<type:historical>",
            "experimental": "<type:experimental>",
        }

        self.models = {
            # --- 民航客机 (Airliners) ---
            "a300": "<model:airbus-a300>", "a310": "<model:airbus-a310>",
            "a318": "<model:airbus-a318>", "a319": "<model:airbus-a319>",
            "a320": "<model:airbus-a320>", "a321": "<model:airbus-a321>",
            "a320neo": "<model:airbus-a320neo>", "a321neo": "<model:airbus-a321neo>",
            "a330": "<model:airbus-a330>", "a330neo": "<model:airbus-a330neo>",
            "a340": "<model:airbus-a340>",
            "a350_900": "<model:airbus-a350-900>", "a350_1000": "<model:airbus-a350-1000>",
            "a380": "<model:airbus-a380>", "a220": "<model:airbus-a220>",
            "b707": "<model:boeing-707>", "b717": "<model:boeing-717>",
            "b727": "<model:boeing-727>", "b737": "<model:boeing-737>",
            "b737_max": "<model:boeing-737-max>",
            "b747": "<model:boeing-747>", "b747_8": "<model:boeing-747-8>",
            "b757": "<model:boeing-757>", "b767": "<model:boeing-767>",
            "b777": "<model:boeing-777>", "b777x": "<model:boeing-777x>",
            "b787": "<model:boeing-787>", "md_11": "<model:mcdonnell-douglas-md-11>",
            "md_80": "<model:mcdonnell-douglas-md-80>",
            "l1011": "<model:lockheed-l-1011-tristar>",
            "concorde": "<model:aerospatiale-bac-concorde>",
            "c919": "<model:comac-c919>", "arj21": "<model:comac-arj21>",
            "crj900": "<model:bombardier-crj900>", "embraer_e190": "<model:embraer-e190>",
            "atr_72": "<model:atr-72>",

            # --- 货机 (Cargo) ---
            "b747_freighter": "<model:boeing-747-freighter>",
            "a330_freighter": "<model:airbus-a330-freighter>",
            "an124": "<model:antonov-an-124>", "an225": "<model:antonov-an-225>",
            "beluga": "<model:airbus-beluga>", "dreamlifter": "<model:boeing-dreamlifter>",

            # --- 无人机 (UAVs) ---
            "dji_mavic": "<model:dji-mavic-series>", "dji_phantom": "<model:dji-phantom-series>",
            "dji_inspire": "<model:dji-inspire-series>", "dji_agras": "<model:dji-agras>",
            "wing_loong": "<model:chengdu-wing-loong>", "ch_5": "<model:caihong-5>",
            "mq_9_reaper": "<model:ga-mq-9-reaper>", "rq_4_global_hawk": "<model:northrop-grumman-rq-4>",

            # --- 军机 (Military) ---
            "f22_raptor": "<model:lockheed-f-22-raptor>", "f35_lightning_ii": "<model:lockheed-f-35-lightning-ii>",
            "j20": "<model:chengdu-j-20>", "su57": "<model:sukhoi-su-57>",
            "f16_fighting_falcon": "<model:general-dynamics-f-16>", "f18_hornet": "<model:mcdonnell-douglas-f-a-18>",
            "b2_spirit": "<model:northrop-grumman-b-2-spirit>", "b52_stratofortress": "<model:boeing-b-52>",
            "h20": "<model:xian-h-20>", "c17_globemaster": "<model:boeing-c-17>",
            "c5_galaxy": "<model:lockheed-c-5-galaxy>", "y20": "<model:xian-y-20>",
            "a10_warthog": "<model:fairchild-republic-a-10>",

            # --- 直升机 (Helicopters) ---
            "ah64_apache": "<model:boeing-ah-64-apache>", "uh60_black_hawk": "<model:sikorsky-uh-60>",
            "ch47_chinook": "<model:boeing-ch-47>", "z20": "<model:harbin-z-20>",
            "bell_206": "<model:bell-206>", "robinson_r44": "<model:robinson-r44>",

            # --- 航天器 (Spacecraft) ---
            "falcon_9": "<model:spacex-falcon-9>", "falcon_heavy": "<model:spacex-falcon-heavy>",
            "starship": "<model:spacex-starship>", "long_march_5": "<model:long-march-5>",
            "long_march_9": "<model:long-march-9>",
            "space_shuttle": "<model:nasa-space-shuttle>", "soyuz": "<model:soyuz-spacecraft>",
            "shenzhou": "<model:shenzhou-spacecraft>", "iss": "<model:iss>", "tiangong_station": "<model:tiangong-space-station>",
            "james_webb_telescope": "<model:jwst>", "hubble_telescope": "<model:hubble-space-telescope>",
            "voyager_probe": "<model:voyager-probe>",
        }

        self.components = {
            # --- 外部通用 ---
            "fuselage": "<component:fuselage>", "wing": "<component:wing>",
            "engine": "<component:engine>", "cockpit_windows": "<component:cockpit-windows>",
            "passenger_windows": "<component:passenger-windows>", "tail_fin": "<component:tail-fin>",
            "landing_gear": "<component:landing-gear>", "nose_cone": "<component:nose-cone>",
            "radome": "<component:radome>", "antenna": "<component:antenna>",
            "pitot_tube": "<component:pitot-tube>", "static_port": "<component:static-port>",
            "door": "<component:door>", "emergency_exit": "<component:emergency-exit>",

            # --- 飞控面 ---
            "flaps": "<component:flaps>", "slats": "<component:slats>",
            "aileron": "<component:aileron>", "spoiler": "<component:spoiler>",
            "elevator": "<component:elevator>", "rudder": "<component:rudder>",
            "winglet": "<component:winglet>", "sharklet": "<component:sharklet>",
            "raked_wingtip": "<component:raked-wingtip>", "canard": "<component:canard>",

            # --- 动力系统 ---
            "turbofan_engine": "<component:engine-turbofan>", "turboprop_engine": "<component:engine-turboprop>",
            "piston_engine": "<component:engine-piston>", "jet_engine": "<component:engine-jet>",
            "engine_inlet": "<component:engine-inlet>", "engine_fan_blades": "<component:engine-fan-blades>",
            "engine_nacelle": "<component:engine-nacelle>", "engine_exhaust": "<component:engine-exhaust>",
            "thrust_reverser": "<component:thrust-reverser>", "apu": "<component:apu>",
            "propeller": "<component:propeller>",

            # --- 起落架系统 ---
            "nose_gear": "<component:nose-gear>", "main_gear": "<component:main-gear>",
            "gear_strut": "<component:gear-strut>", "wheel": "<component:wheel>", "tire": "<component:tire>",

            # --- 驾驶舱内部 ---
            "instrument_panel": "<component:instrument-panel>", "pfd": "<component:pfd>",
            "mfd": "<component:mfd>",
            "ecam_eicas": "<component:ecam-eicas>",
            "fmc_mcdu": "<component:fmc-mcdu>",
            "yoke_sidestick": "<component:yoke-sidestick>", "throttle_levers": "<component:throttle-levers>",
            "overhead_panel": "<component:overhead-panel>", "rudder_pedals": "<component:rudder-pedals>",
            "hud": "<component:hud>",

            # --- 军用/特殊 ---
            "weapon_pylon": "<component:weapon-pylon>", "missile": "<component:missile>",
            "bomb": "<component:bomb>", "gun_pod": "<component:gun-pod>",
            "targeting_pod": "<component:targeting-pod>", "sensor_turret": "<component:sensor-turret>",
            "air_refueling_boom": "<component:air-refueling-boom>", "arresting_hook": "<component:arresting-hook>",
            "stealth_shaping": "<component:stealth-shaping>",

            # --- 直升机 ---
            "main_rotor": "<component:main-rotor>", "tail_rotor": "<component:tail-rotor>", "skids": "<component:skids>",

            # --- 航天器 ---
            "booster": "<component:booster>", "first_stage": "<component:first-stage>",
            "second_stage": "<component:second-stage>", "interstage": "<component:interstage>",
            "fairing": "<component:fairing>", "rocket_nozzle": "<component:rocket-nozzle>",
            "grid_fin": "<component:grid-fin>", "solar_panel": "<component:solar-panel>",
            "payload_bay": "<component:payload-bay>", "docking_port": "<component:docking-port>",
            "heat_shield": "<component:heat-shield>",
        }

        self.states = {
            # --- 飞行阶段 ---
            "phase_parked": "<phase:parked>", "phase_pushback": "<phase:pushback>",
            "phase_taxiing": "<phase:taxiing>", "phase_takeoff_roll": "<phase:takeoff-roll>",
            "phase_rotation": "<phase:rotation>", "phase_initial_climb": "<phase:initial-climb>",
            "phase_climb": "<phase:climb>", "phase_cruise": "<phase:cruise>",
            "phase_descent": "<phase:descent>", "phase_approach": "<phase:approach>",
            "phase_final_approach": "<phase:final-approach>", "phase_flare": "<phase:flare>",
            "phase_landing_rollout": "<phase:landing-rollout>",
            "phase_go_around": "<phase:go-around>",
            
            # --- 操作与构型 ---
            "op_flaps_retracted": "<op:flaps-retracted>", "op_flaps_takeoff": "<op:flaps-takeoff>",
            "op_flaps_landing": "<op:flaps-landing>", "op_flaps_in_motion": "<op:flaps-in-motion>",
            "op_gear_up": "<op:gear-up>", "op_gear_down": "<op:gear-down>",
            "op_gear_in_transit": "<op:gear-in-transit>",
            "op_spoilers_retracted": "<op:spoilers-retracted>", "op_spoilers_armed": "<op:spoilers-armed>",
            "op_spoilers_deployed": "<op:spoilers-deployed>",
            "op_reversers_stowed": "<op:reversers-stowed>", "op_reversers_deployed": "<op:reversers-deployed>",
            "op_engine_startup": "<op:engine-startup>", "op_engine_shutdown": "<op:engine-shutdown>",
            "op_lights_nav_on": "<op:lights-nav-on>", "op_lights_landing_on": "<op:lights-landing-on>",
            
            # --- 航天事件 ---
            "event_liftoff": "<event:liftoff>", "event_max_q": "<event:max-q>",
            "event_booster_separation": "<event:booster-separation>",
            "event_stage_separation": "<event:stage-separation>",
            "event_fairing_jettison": "<event:fairing-jettison>",
            "event_entry_burn": "<event:entry-burn>", "event_landing_burn": "<event:landing-burn>",
            "event_touchdown": "<event:touchdown>", "event_docking": "<event:docking>",
            "event_undocking": "<event:undocking>", "event_payload_deploy": "<event:payload-deploy>",
            
            # --- 维护与特殊状态 ---
            "state_maintenance": "<state:maintenance>", "state_storage": "<state:storage>",
            "state_scrapped": "<state:scrapped>", "state_damaged": "<state:damaged>",
            "state_emergency": "<state:emergency>", # e.g., engine fire
            "state_bird_strike": "<state:bird-strike>", "state_icing": "<state:icing>",
        }

        self.attributes = {
            # --- 视角 ---
            "view_side": "<view:side>", "view_front": "<view:front>", "view_rear": "<view:rear>",
            "view_top_down": "<view:top-down>", "view_bottom_up": "<view:bottom-up>",
            "view_cockpit": "<view:cockpit>", "view_onboard": "<view:onboard>",
            "view_wing": "<view:wing>", "view_passenger": "<view:passenger>",
            "view_close_up": "<view:close-up>", "view_wide_angle": "<view:wide-angle>",
            "view_telephoto": "<view:telephoto>", "view_dutch_angle": "<view:dutch-angle>",
            
            # --- 环境 ---
            "env_day": "<env:day>", "env_night": "<env:night>", "env_dawn": "<env:dawn>", "env_dusk": "<env:dusk>",
            "weather_clear": "<weather:clear>", "weather_cloudy": "<weather:cloudy>",
            "weather_overcast": "<weather:overcast>", "weather_rain": "<weather:rain>",
            "weather_snow": "<weather:snow>", "weather_fog": "<weather:fog>",
            "ground_runway": "<ground:runway>", "ground_taxiway": "<ground:taxiway>",
            "ground_apron": "<ground:apron>", "ground_grass": "<ground:grass>",
            "location_airport": "<location:airport>", "location_airshow": "<location:airshow>",
            "location_in_flight": "<location:in-flight>",
            
            # --- 涂装与外观 ---
            "livery_standard": "<livery:standard>", "livery_special": "<livery:special>",
            "livery_retro": "<livery:retro>", "livery_alliance": "<livery:alliance>", # e.g., Star Alliance
            "livery_military_camo": "<livery:military-camo>", "livery_none": "<livery:none>", # e.g., primer
            "condition_clean": "<condition:clean>", "condition_dirty": "<condition:dirty>",
            "condition_weathered": "<condition:weathered>", "condition_damaged": "<condition:damaged>",
            "effect_vortex": "<effect:vortex>", "effect_contrail": "<effect:contrail>",
            "effect_heat_haze": "<effect:heat-haze>", "effect_vapor_cone": "<effect:vapor-cone>",
        }

    def get_all_tokens(self) -> list[str]:
        all_tokens = set()
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, dict):
                all_tokens.update(attr_value.values())
        return sorted(list(all_tokens))
      
if __name__ == "__main__":
    vocab = SpecialTokens()
    all_special_tokens = vocab.get_all_tokens()

    print(f"--- Total {len(all_special_tokens)} Special Tokens Generated ---")
    if len(all_special_tokens) > 20:
        print("First 10 tokens:", all_special_tokens[:10])
        print("Last 10 tokens:", all_special_tokens[-10:])
    else:
        print(all_special_tokens)
    print("\n" + "="*50 + "\n")

    # 构建复杂描述性序列的例子

    # 1. 一架身披特殊涂装的波音747-8在雨天跑道上进行起飞加速，襟翼已设定，可以看到机翼涡流
    b747_takeoff_desc = (
        f"{vocab.categories['airliner']} {vocab.models['b747_8']} "
        f"{vocab.states['phase_takeoff_roll']} {vocab.states['op_flaps_takeoff']} {vocab.states['op_gear_down']} "
        f"{vocab.attributes['ground_runway']} {vocab.attributes['weather_rain']} "
        f"{vocab.attributes['livery_special']} {vocab.attributes['effect_vortex']}"
    )
    print("Example 1: B747 Takeoff Roll in Rain")
    print(b747_takeoff_desc)
    print("-" * 20)

    # 2. 从驾驶舱视角看，一架F-22猛禽战斗机正在进行空中加油，视角为广角
    f22_refueling_desc = (
        f"{vocab.categories['military_fighter']} {vocab.models['f22_raptor']} "
        f"{vocab.components['air_refueling_boom']} {vocab.attributes['view_cockpit']} "
        f"{vocab.attributes['view_wide_angle']} {vocab.attributes['location_in_flight']}"
    )
    print("Example 2: F-22 Air Refueling (Cockpit View)")
    print(f22_refueling_desc)
    print("-" * 20)

    # 3. 长征五号火箭成功实现级间分离的遥摄图像
    lm5_separation_desc = (
        f"{vocab.categories['spacecraft']} {vocab.models['long_march_5']} "
        f"{vocab.states['event_stage_separation']} {vocab.components['first_stage']} "
        f"{vocab.components['second_stage']} {vocab.attributes['view_telephoto']}"
    )
    print("Example 3: Long March 5 Stage Separation")
    print(lm5_separation_desc)
    print("-" * 20)
    
    # 4. 一架Robinson R44直升机在机库内进行维护，主旋翼被固定
    r44_maintenance_desc = (
        f"{vocab.categories['helicopter']} {vocab.models['robinson_r44']} "
        f"{vocab.states['state_maintenance']} {vocab.components['main_rotor']} "
        f"{vocab.attributes['ground_apron']}" # Assuming inside or near a hangar on the apron
    )
    print("Example 4: Helicopter Maintenance")
    print(r44_maintenance_desc)
    print("-" * 20)
