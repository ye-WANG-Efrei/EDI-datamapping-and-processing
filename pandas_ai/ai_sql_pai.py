import pandasai as pai

# Get your API key from https://app.pandabi.ai
pai.api_key.set("PAI-a0a29f88-52fa-4c0b-9629-be92f5a4431f")

df = pai.read_excel("data/employee_data.xlsx")

response = df.chat("What is the average revenue by region?")
print(response)