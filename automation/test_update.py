import os
import clr
import System

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

# ----------------------------
# Create Automation Object
# ----------------------------
automationAssembly = System.Reflection.Assembly.Load("DWSIM.Automation")
Automation3 = automationAssembly.GetType("DWSIM.Automation.Automation3")
sim = System.Activator.CreateInstance(Automation3)

flowsheet_path = r"C:\Users\Dell\OneDrive\Desktop\DWSIM_Surrogate_Modelling_ML\dwsim_files\Benzene_Toluene.dwxmz"

print("Loading flowsheet for testing...")
flowsheet = sim.LoadFlowsheet(flowsheet_path)

feed = flowsheet.GetFlowsheetSimulationObject("Feed").GetAsObject()

print("--- BEFORE CHANGE ---")
print("Feed Temperature:", feed.GetTemperature())

# Test setting a new temperature
test_temp = 365.0
feed.SetTemperature(test_temp)

print(f"\nSetting Feed Temperature to {test_temp} K and solving...")

flowsheet.ResetCalculationStatus()
errors = flowsheet.RequestCalculationAndWait()

print("Calculation Errors:", errors.Count)

# Fetch outputs
distillate = flowsheet.GetFlowsheetSimulationObject("3").GetAsObject()
bottoms = flowsheet.GetFlowsheetSimulationObject("4").GetAsObject()

print("\n--- AFTER CHANGE ---")
print("New Feed Temperature   :", feed.GetTemperature())
print("Resulting Distillate T :", distillate.GetTemperature())
print("Resulting Bottoms T    :", bottoms.GetTemperature())

print("\nTest Complete!")