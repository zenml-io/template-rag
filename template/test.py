from zenml import step, pipeline


@step
def my_step():
    print("Hello, world!")


@pipeline
def my_pipeline():
    my_step()


if __name__ == "__main__":
    my_pipeline()
