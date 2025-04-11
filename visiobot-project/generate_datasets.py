import os

# Create a directory to store the CSV files
output_dir = "visio_datasets"
os.makedirs(output_dir, exist_ok=True)

# 1. Histogram dataset
histogram_csv = os.path.join(output_dir, "histogram_data.csv")
with open(histogram_csv, "w") as f:
    f.write("""Score
85
90
85
92
88
76
95
80
83
89
85
90
92
76
""")

# 2. Line Chart dataset
line_csv = os.path.join(output_dir, "line_data.csv")
with open(line_csv, "w") as f:
    f.write("""Date,Sales
2023-01-01,150
2023-01-02,180
2023-01-03,160
2023-01-04,200
2023-01-05,190
2023-01-06,210
2023-01-07,205
""")

# 3. Linked Graph dataset
linked_csv = os.path.join(output_dir, "linked_data.csv")
with open(linked_csv, "w") as f:
    f.write("""Source,Target
A,B
B,C
C,D
D,E
E,A
""")

# 4. Map dataset
map_csv = os.path.join(output_dir, "map_data.csv")
with open(map_csv, "w") as f:
    f.write("""Country,GDP (BILLIONS),Code
USA,21000,US
China,14000,CN
Germany,4000,DE
India,2800,IN
Brazil,1900,BR
""")

# 5. Parallel Coordinates dataset
parallel_csv = os.path.join(output_dir, "parallel_data.csv")
with open(parallel_csv, "w") as f:
    f.write("""Feature1,Feature2,Feature3,Group
5.5,3.2,1.4,A
6.0,3.0,4.5,B
5.7,3.8,1.7,A
6.3,2.9,5.6,B
5.8,3.0,1.2,A
6.5,3.0,4.8,B
5.9,3.0,1.8,A
""")

# 6. Pie Chart dataset
pie_csv = os.path.join(output_dir, "pie_data.csv")
with open(pie_csv, "w") as f:
    f.write("""Category,Value
Apples,50
Bananas,30
Cherries,20
Grapes,40
""")

# 7. Scatter Plot dataset
scatter_csv = os.path.join(output_dir, "scatter_data.csv")
with open(scatter_csv, "w") as f:
    f.write("""X,Y
1.1,2.3
2.2,3.5
3.3,4.7
4.4,6.1
5.0,7.2
""")

# 8. Treemap dataset
treemap_csv = os.path.join(output_dir, "treemap_data.csv")
with open(treemap_csv, "w") as f:
    f.write("""MainCategory,SubCategory,Value
Fruits,Citrus,40
Fruits,Berries,30
Vegetables,Leafy,20
Vegetables,Root,10
Grains,Cereals,50
Grains,Legumes,25
""")

print("CSV files have been created in the directory:", output_dir)
