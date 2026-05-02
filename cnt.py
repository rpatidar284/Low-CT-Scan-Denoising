import os

# Define the root directory
data_root = "data"

def count_images(folder, extensions=('.png', '.jpg', '.jpeg', '.dcm', '.tiff')):
    """Counts files in a folder that match specific image extensions."""
    count = 0
    if not os.path.exists(folder):
        return 0
        
    for root, _, files in os.walk(folder):
        for f in files:
            # Check if the file ends with any of the allowed extensions (case-insensitive)
            if f.lower().endswith(extensions):
                count += 1
    return count

def main():
    if not os.path.exists(data_root):
        print(f"Error: The path '{data_root}' does not exist.")
        return

    total_hdct = 0
    total_ldct = 0

    print(f"{'Patient ID':<15} | {'HDCT':<10} | {'LDCT':<10}")
    print("-" * 40)

    # Sort patient names to keep the output organized
    for patient in sorted(os.listdir(data_root)):
        patient_path = os.path.join(data_root, patient)

        # Skip files in the root (like .txt or .csv files)
        if not os.path.isdir(patient_path):
            continue

        hdct_path = os.path.join(patient_path, "HDCT")
        ldct_path = os.path.join(patient_path, "LDCT")

        # Get counts
        hdct_count = count_images(hdct_path)
        ldct_count = count_images(ldct_path)

        # Update grand totals
        total_hdct += hdct_count
        total_ldct += ldct_count

        print(f"{patient:<15} | {hdct_count:<10} | {ldct_count:<10}")

    print("-" * 40)
    print(f"{'TOTAL':<15} | {total_hdct:<10} | {total_ldct:<10}")

if __name__ == "__main__":
    main()

# Patient ID      | HDCT       | LDCT      
# ----------------------------------------
# C002            | 280        | 280       
# C004            | 361        | 361       
# C012            | 351        | 351       
# C016            | 319        | 319       
# C021            | 378        | 378       
# C030            | 303        | 303       
# C050            | 394        | 394       
# C052            | 342        | 342       
# C067            | 365        | 365       
# C081            | 356        | 356       
# C095            | 322        | 322       
# C107            | 391        | 391       
# C121            | 392        | 392       
# C124            | 383        | 383       
# C128            | 345        | 345       
# C130            | 379        | 379       
# C135            | 355        | 355       
# C158            | 363        | 363       
# C160            | 315        | 315       
# C162            | 343        | 343       
# C179            | 334        | 334       
# C193            | 365        | 365       
# C203            | 302        | 302       
# C218            | 357        | 357       
# C219            | 344        | 344       
# C224            | 332        | 332       
# C227            | 303        | 303       
# C232            | 353        | 353       
# C234            | 380        | 380       
# C241            | 366        | 366       
# C246            | 363        | 363       
# C252            | 363        | 363       
# C261            | 329        | 329       
# C267            | 333        | 333       
# C268            | 370        | 370       
# C280            | 369        | 369       
# C295            | 303        | 303       
# C296            | 329        | 329       
# L004            | 99         | 99        
# L006            | 215        | 215       
# L019            | 169        | 169       
# L033            | 162        | 162       
# L056            | 93         | 93        
# L058            | 210        | 210       
# L064            | 209        | 209       
# L071            | 154        | 154       
# L072            | 142        | 142       
# L077            | 161        | 161       
# L081            | 132        | 132       
# L110            | 133        | 133       
# L123            | 151        | 151       
# L125            | 205        | 205       
# L131            | 91         | 91        
# L134            | 217        | 217       
# L145            | 160        | 160       
# L148            | 210        | 210       
# L160            | 115        | 115       
# L170            | 137        | 137       
# L178            | 202        | 202       
# L179            | 143        | 143       
# L186            | 167        | 167       
# L193            | 169        | 169       
# L209            | 98         | 98        
# L210            | 149        | 149       
# L212            | 204        | 204       
# L220            | 156        | 156       
# L221            | 99         | 99        
# L237            | 140        | 140       
# L241            | 155        | 155       
# L266            | 175        | 175       
# masks           | 0          | 0         
# ----------------------------------------
# TOTAL           | 18254      | 18254 