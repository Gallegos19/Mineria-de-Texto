from modules.text_mining_system import TextMiningSystem

def test_system():
    """Test the text mining system with a simple example"""
    # Create an instance of the system
    system = TextMiningSystem()
    
    # Test text
    test_text = "El reciclaje es importante para el planeta. Debemos cuidar el medio ambiente."
    
    # Process the text
    try:
        print("Processing text...")
        result = system.process_text_complete_enhanced(
            text=test_text,
            content_type='contenido',
            track_steps=True
        )
        
        # Print the result
        print("\nOriginal text:", result['original_text'])
        print("Final text:", result['final_text'])
        print("\nIntermediate results:")
        for step, text in result['intermediate_results'].items():
            print(f"- {step}: {text}")
        
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()