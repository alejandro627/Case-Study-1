"""
Discrete-event simulation (SimPy) of the wooden-box production and assembly line in its reference configuration.

Model scope
-----------
- 8-hour working shift (28,800 s)
- 3 stages:
    1) CNC laser machining
    2) collaborative assembly workstation
    3) quality control and packaging

Modelling assumptions
---------------------
1. One CNC machine, one collaborative assembly station and one QC/packaging station, each with one operator.
2. Raw material is limited to the initial stock of RAW_PANELS = 30 plywood panels.
3. One CNC cycle (setup + machining) turns one plywood panel into PANELS_PER_BOX = 6 wooden panels, i.e. one box kit.
4. Assembly consumes 6 wooden panels and 8 angular elements per box.
5. QC inspects the boxes one by one; every BOXES_PER_CARTON = 2 approved boxes form one carton.
6. Process times are deterministic means of three measured values; no defects are modelled.
7. The simulation stops at 8 h; jobs waiting or in progress remain in their current state.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

import simpy

# =============================
# SIMULATION SETTINGS
# =============================
SHIFT_HOURS = 8
SIM_TIME = SHIFT_HOURS * 3600

# Deterministic mean times
CNC_SETUP_MEAN = (240 + 270 + 300)/3
CNC_PROCESS_MEAN = (762 + 766 + 770)/3

ASM_SETUP_MEAN = (30 + 45 + 60)/3
ASM_PROCESS_MEAN = (420 + 450 + 480)/3

QC_SETUP_MEAN = (10 + 15 + 20)/3
QC_PROCESS_MEAN = (240 + 270 + 300)/3

# System parameters
RAW_PANELS = 30
PANELS_PER_BOX = 6
BOXES_PER_CARTON = 2
ANGULAR_PER_BOX = 8

# Energy parameters [kW]
P_CNC = 1.9 # CNC machine power consumption
P_COBOT = 0.38 # Collaborative robot power consumption
P_QC = 0.10 # Quality control station power consumption

# =============================
# METRICS
# =============================
class Metrics:
    def __init__(self):
        self.cartons_completed = 0
        self.boxes_produced = 0
        self.panels_produced = 0
        self.qc_boxes = 0

        self.angular_consumed = 0
        self.cardboard_consumed = 0

        self.busy_time_cnc = 0
        self.busy_time_cobot = 0
        self.busy_time_op3 = 0

        # WIP tracking - kept per-stage (panels vs. boxes are different
        # physical units, PANELS_PER_BOX apart; summing them into one number
        # would silently mix units - see update_wip() below).
        self.panel_wip_time_integral = 0
        self.box_wip_time_integral = 0
        self.last_event_time = 0

        # True per-unit flow-time tracking (distinct from the pacing-based
        # "3600/TH" cycle time computed in main()). box_flow_times[i] is the
        # time one specific box's raw material spent in the system, from the
        # instant its plywood panel left raw_panels (CNC start) to the
        # instant that box cleared QC. carton_flow_times[i] is the analogous
        # quantity for a carton, bounded by whichever of its
        # BOXES_PER_CARTON contributing boxes entered the system LAST (a
        # carton cannot be completed before its slowest-arriving component
        # is ready - the standard convention for a converging/assembly point).
        self.box_flow_times: List[float] = []
        self.carton_flow_times: List[float] = []

# =============================
# SYSTEM
# =============================
class System:
    def __init__(self, env):
        self.env = env
        self.m = Metrics()

        # Resources
        self.cnc = simpy.Resource(env, capacity=1)
        self.cobot = simpy.Resource(env, capacity=1)
        self.op3 = simpy.Resource(env, capacity=1)

        # Buffers
        self.raw_panels = simpy.Container(env, init=RAW_PANELS)
        self.panel_buffer = simpy.Container(env, init=0)
        self.box_buffer = simpy.Container(env, init=0)

        # FIFO queues of "genesis" timestamps, carried alongside the
        # quantity-only Containers above, used only for flow-time tracking.
        # Safe as plain FIFOs because each buffer has exactly one producer
        # and one consumer process, so arrival order is preserved and each
        # get() always corresponds to the oldest not-yet-consumed batch.
        self.panel_release_times: deque[float] = deque()  # one entry per completed CNC batch (= one future box)
        self.box_release_times: deque[float] = deque()    # one entry per completed box, carrying its panel batch's release time forward

    # --------------------------
    # WIP TRACKING
    # --------------------------
    def update_wip(self):
        # Tracked separately per stage, not summed: panel_buffer.level is in
        # panels, box_buffer.level is in boxes, and PANELS_PER_BOX panels
        # convert to only 1 box - adding the raw counts would over-weight
        # panel-stage WIP relative to box-stage WIP by that factor.
        dt = self.env.now - self.m.last_event_time
        self.m.panel_wip_time_integral += self.panel_buffer.level * dt
        self.m.box_wip_time_integral += self.box_buffer.level * dt
        self.m.last_event_time = self.env.now

    # --------------------------
    # CNC PROCESS
    # --------------------------
    def cnc_process(self):
        while self.raw_panels.level > 0:

            with self.cnc.request() as req:
                yield req

                yield self.raw_panels.get(1)

                # Raw material enters the production system now - this batch's
                # flow-time clock starts here, not when the box is completed.
                release_time = self.env.now

                time = CNC_SETUP_MEAN + CNC_PROCESS_MEAN
                start = self.env.now

                yield self.env.timeout(time)

                self.m.busy_time_cnc += self.env.now - start

                yield self.panel_buffer.put(PANELS_PER_BOX)
                self.panel_release_times.append(release_time)

                self.m.panels_produced += PANELS_PER_BOX
                self.update_wip()

    # --------------------------
    # ASSEMBLY
    # --------------------------
    def assembly_process(self):
        while True:

            if self.panel_buffer.level < PANELS_PER_BOX:
                yield self.env.timeout(1)
                continue

            with self.cobot.request() as req:
                yield req

                yield self.panel_buffer.get(PANELS_PER_BOX)
                # PANELS_PER_BOX is exactly one CNC batch's output, so this
                # always corresponds to the oldest not-yet-consumed batch.
                release_time = self.panel_release_times.popleft()

                time = ASM_SETUP_MEAN + ASM_PROCESS_MEAN
                start = self.env.now

                yield self.env.timeout(time)

                self.m.busy_time_cobot += self.env.now - start

                yield self.box_buffer.put(1)
                self.box_release_times.append(release_time)

                self.m.boxes_produced += 1
                self.m.angular_consumed += ANGULAR_PER_BOX
                self.m.box_flow_times.append(self.env.now - release_time)

                self.update_wip()

    # --------------------------
    # QC + PACKAGING
    # --------------------------
    def qc_process(self):
        while True:

            if self.box_buffer.level < BOXES_PER_CARTON:
                yield self.env.timeout(1)
                continue

            with self.op3.request() as req:
                yield req

                yield self.box_buffer.get(BOXES_PER_CARTON)
                # A carton cannot start before its slowest-arriving
                # contributing box entered the system - use the max, not
                # the mean, of the consumed boxes' own genesis times.
                contributing_release_times = [
                    self.box_release_times.popleft() for _ in range(BOXES_PER_CARTON)
                ]
                carton_release_time = max(contributing_release_times)

                time = QC_SETUP_MEAN + QC_PROCESS_MEAN
                start = self.env.now

                yield self.env.timeout(time)

                self.m.busy_time_op3 += self.env.now - start

                self.m.qc_boxes += BOXES_PER_CARTON
                self.m.cartons_completed += 1
                self.m.cardboard_consumed += 1
                self.m.carton_flow_times.append(self.env.now - carton_release_time)

                self.update_wip()

# =============================
# MAIN
# =============================
def main():
    env = simpy.Environment()
    sys = System(env)

    env.process(sys.cnc_process())
    env.process(sys.assembly_process())
    env.process(sys.qc_process())

    env.run(until=SIM_TIME)

    # =============================
    # KPI CALCULATION
    # =============================
    throughput = sys.m.cartons_completed / (SIM_TIME / 3600)
    avg_wip_panels = sys.m.panel_wip_time_integral / SIM_TIME
    avg_wip_boxes = sys.m.box_wip_time_integral / SIM_TIME

    # Two distinct notions of "Cycle Time" - see the docstrings in
    # cnc_process/assembly_process/qc_process for how the flow-time ones are
    # tracked:
    #
    # 1) CT_pace: average time-between-completions (= SIM_TIME / count =
    #    algebraically identical to 3600/throughput). Useful for pacing /
    #    capacity questions ("how many seconds of shift does one carton
    #    correspond to"), but NOT the time any single carton actually spent
    #    in the system - by Little's Law it equals the true flow time only
    #    if avg WIP = 1, which is not the case here (CNC is the bottleneck,
    #    so material queues up behind it).
    # 2) CT_flow_*: true per-unit sojourn time, measured from when that
    #    unit's raw material left raw_panels to when it cleared QC -
    #    genuinely tracked per box/per carton, then averaged.
    CT_pace = SIM_TIME / sys.m.cartons_completed if sys.m.cartons_completed > 0 else float('inf')
    CT_flow_box = (
        sum(sys.m.box_flow_times) / len(sys.m.box_flow_times)
        if sys.m.box_flow_times else float('inf')
    )
    CT_flow_carton = (
        sum(sys.m.carton_flow_times) / len(sys.m.carton_flow_times)
        if sys.m.carton_flow_times else float('inf')
    )

    # Energy
    E_CNC = P_CNC * sys.m.busy_time_cnc / 3600
    E_A = P_COBOT * sys.m.busy_time_cobot / 3600
    E_QC = P_QC * sys.m.busy_time_op3 / 3600
    E_tot = E_CNC + E_A + E_QC

    # Utilization
    def util(x): return 100 * x / SIM_TIME

    # =============================
    # REPORT
    # =============================
    print("\n========== RESULTS ==========")

    print("\n--- PRODUCTION ---")
    print(f"Raw panels consumed (Plywood panels): {RAW_PANELS - sys.raw_panels.level}")
    print(f"Machined wooden panels total: {sys.m.panels_produced}")
    print(f"Assembled boxes total: {sys.m.boxes_produced}")
    print(f"QC boxes inspected: {sys.m.qc_boxes}")
    print(f"Cartons produced: {sys.m.cartons_completed}")

    print("\n--- CONSUMPTION ---")
    print(f"Angular elements consumed: {sys.m.angular_consumed}")
    print(f"Cardboard cartons consumed: {sys.m.cardboard_consumed}")

    print("\n--- INVENTORIES ---")
    print(f"Raw panels remaining: {sys.raw_panels.level}")
    print(f"Panels in buffer: {sys.panel_buffer.level}")
    print(f"Boxes in buffer: {sys.box_buffer.level}")

    print("\n--- RESOURCE UTILIZATION ---")
    print(f"CNC utilization: {util(sys.m.busy_time_cnc):.2f}%")
    print(f"Cobot utilization: {util(sys.m.busy_time_cobot):.2f}%")
    print(f"Operator 3 utilization: {util(sys.m.busy_time_op3):.2f}%")

    print("\n--- KPIs ---")
    print(f"Throughput [cartons/h]: {throughput:.3f}")
    print(f"Average WIP - panels in buffer [panels]: {avg_wip_panels:.3f}")
    print(f"Average WIP - boxes in buffer [boxes]: {avg_wip_boxes:.3f}")
    print(f"Cycle Time - pacing, 3600/TH [s/carton]: {CT_pace:.2f}")
    print(f"Cycle Time - true flow time, per box [s/box]: {CT_flow_box:.2f}  (n={len(sys.m.box_flow_times)})")
    print(f"Cycle Time - true flow time, per carton [s/carton]: {CT_flow_carton:.2f}  (n={len(sys.m.carton_flow_times)})")

    print("\n--- ENERGY ---")
    print(f"E_CNC [kWh]: {E_CNC:.2f}")
    print(f"E_A [kWh]: {E_A:.2f}")
    print(f"E_QC [kWh]: {E_QC:.2f}")
    print(f"E_tot [kWh]: {E_tot:.2f}")

if __name__ == "__main__":
    main()
