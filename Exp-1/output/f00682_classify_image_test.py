from f00682_classify_image import *
categories = ['animals','vegetables', 'city landscape', 'cars', 'office']
image_url = "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80"

result = classify_image(categories, image_url)
print(result)

