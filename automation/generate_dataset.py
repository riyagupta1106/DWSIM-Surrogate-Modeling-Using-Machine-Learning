import os
import clr
import System
import random
import csv

# ----------------------------
# DWSIM PATH
# ----------------------------
DWSIM_PATH = r"C:\Users\Dell\AppData\Local\DWSIM"
os.chdir(DWSIM_PATH)

# ----------------------------
# Load DLLs
# ----------------------------
clr.AddReference(os.path.join(DWSIM_PATH, "CapeOpen.dll"))
clr.AddReference(os.path.join(DWSIM_PATH, "DWSIM.Interfaces.dll"))
clr.AddReference(r"C:\Users\Dell\AppData\Local\DWSIM\ThermoCS\ThermoCS.dll")
clr.AddReference(os.path.join(DWSIM_PATH, "DWSIM.Automation.dll"))

print("DLLs Loaded Successfully")

# ----------------------------
# Create Automation Object
# ----------------------------
automationAssembly = System.Reflection.Assembly.Load("DWSIM.Automation")
Automation3 = automationAssembly.GetType("DWSIM.Automation.Automation3")
sim = System.Activator.CreateInstance(Automation3)

flowsheet_path = r"C:\Users\Dell\OneDrive\Desktop\DWSIM_Surrogate_Modelling_ML\dwsim_files\Benzene_Toluene.dwxmz"

rows = []
num_samples = 200

print(f"\nStarting dataset generation for {num_samples} samples...\n")

for i in range(num_samples):
    # Load flowsheet fresh every iteration
    flowsheet = sim.LoadFlowsheet(flowsheet_path)
    
    # Get Feed Object using correct stream name "2" from your flowsheet
    feed = flowsheet.GetFlowsheetSimulationObject("2").GetAsObject()

    # 1. Generate Random Inputs
    T = random.uniform(330, 380)      # Kelvin
    P = 101325                        # Constant (Pa)
    F = random.uniform(0.5, 3.0)      # kg/s
    xB = random.uniform(0.2, 0.8)     # Mole fraction Benzene
    xT = 1.0 - xB                     # Mole fraction Toluene

    # 2. Set Feed Conditions
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMassFlow(F)
    feed.SetOverallComposition(System.Array[float]([xB, xT]))

    # 3. Force calculation
    flowsheet.ResetCalculationStatus()
    errors = flowsheet.RequestCalculationAndWait()

    # 4. Extract Output Streams ("3" for distillate, "4" for bottoms)
    distillate = flowsheet.GetFlowsheetSimulationObject("3").GetAsObject()
    bottoms = flowsheet.GetFlowsheetSimulationObject("4").GetAsObject()

    dist_T = distillate.GetTemperature()
    dist_P = distillate.GetPressure()
    dist_Flow = distillate.GetMassFlow()
    dist_comp = list(distillate.GetOverallComposition())

    bottom_T = bottoms.GetTemperature()
    bottom_P = bottoms.GetPressure()
    bottom_Flow = bottoms.GetMassFlow()
    bottom_comp = list(bottoms.GetOverallComposition())

    print(f"[{i+1}/{num_samples}] Feed T: {T:.2f}K | Dist T: {dist_T:.2f}K | Bottom T: {bottom_T:.2f}K | Errors: {errors.Count}")

    rows.append([
        T, P, F, xB, xT,
        dist_T, dist_P, dist_Flow, dist_comp[0], dist_comp[1],
        bottom_T, bottom_P, bottom_Flow, bottom_comp[0], bottom_comp[1]
    ])

# ----------------------------
# Save Dataset to CSV
# ----------------------------
save_path = os.path.join(
    r"C:\Users\Dell\OneDrive\Desktop\DWSIM_Surrogate_Modelling_ML",
    "data",
    "dataset.csv"
)

os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Feed_T", "Feed_P", "Feed_Flow", "Feed_Benzene", "Feed_Toluene",
        "Dist_T", "Dist_P", "Dist_Flow", "Dist_Benzene", "Dist_Toluene",
        "Bottom_T", "Bottom_P", "Bottom_Flow", "Bottom_Benzene", "Bottom_Toluene"
    ])
    writer.writerows(rows)

print("\nDataset Generation Complete!")
print("Saved at:", save_path)