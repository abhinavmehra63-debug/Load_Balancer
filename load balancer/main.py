from src import data_generator, train_model, simulate, visualize

if __name__ == "__main__":
    print("Step 1: Generating dataset...")
    data_generator.generate_data()

    print("\nStep 2: Training model...")
    train_model.train()

    print("\nStep 3: Simulating load balancer...")
    simulate.simulate_request([30, 50, 15, 100, 250])

    print("\nStep 4: Visualizing results...")
    visualize.plot_server_loads()
    visualize.plot_model_accuracy()
