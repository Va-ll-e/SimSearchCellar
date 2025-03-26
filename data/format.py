
def replace_letter_in_file(file_path, letter_a, letter_b):
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace letter_a with letter_b
        modified_content = content.replace(letter_a, letter_b)
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        return True
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False
    

if __name__ == '__main__':
    replace_letter_in_file('data/weather_data.csv', ';', ',')

    