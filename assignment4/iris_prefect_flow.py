from prefect import flow

@flow
def iris_flow():
    # your preprocessing + training logic
    print("✅ Flow started")
    # return accuracy or model info at the end
    return "Done"

if __name__ == "__main__":
    result = iris_flow()
    print("Flow finished with result:", result)
