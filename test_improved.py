from modules.text_mining_system import TextMiningSystem

def test_improved_system():
    """Test the improved text mining system with problematic examples"""
    # Create an instance of the system
    system = TextMiningSystem()
    
    # Test cases that had issues before
    test_cases = [
        "Cuidemos al planeta",
        "Reciclar es importante",
        "Proteger el medio ambiente",
        "Energía renovable"
    ]
    
    for i, test_text in enumerate(test_cases):
        print(f"\n\n===== TEST CASE {i+1}: '{test_text}' =====")
        
        # Process the text
        try:
            print(f"Processing text: '{test_text}'")
            result = system.process_text_complete_enhanced(
                text=test_text,
                content_type='descripcion',
                track_steps=True
            )
            
            # Print the intermediate results to see the transformation at each step
            print("\nIntermediate results:")
            for step, text in result['intermediate_results'].items():
                print(f"- {step}: '{text}'")
            
            # Print the final result
            print(f"\nFinal text: '{result['final_text']}'")
            
            # Print evaluation metrics
            print("\nEvaluation metrics:")
            for metric, value in result['evaluation'].items():
                print(f"- {metric}: {value}")
            
        except Exception as e:
            print(f"Error during test: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_improved_system()