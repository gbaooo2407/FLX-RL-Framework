import osmnx as ox

# Danh sách quận huyện phổ biến trước đây ở HCM
districts_hcm = [
    "District 1, Ho Chi Minh City, Vietnam",
    "District 2, Ho Chi Minh City, Vietnam",
    "District 3, Ho Chi Minh City, Vietnam",
    "District 4, Ho Chi Minh City, Vietnam",
    "District 5, Ho Chi Minh City, Vietnam",
    "District 6, Ho Chi Minh City, Vietnam",
    "District 7, Ho Chi Minh City, Vietnam",
    "District 8, Ho Chi Minh City, Vietnam",
    "District 9, Ho Chi Minh City, Vietnam",
    "District 10, Ho Chi Minh City, Vietnam",
    "District 11, Ho Chi Minh City, Vietnam",
    "District 12, Ho Chi Minh City, Vietnam",
    "Thu Duc City, Ho Chi Minh City, Vietnam",  # Thành phố Thủ Đức mới
    "Binh Thanh District, Ho Chi Minh City, Vietnam",
    "Phu Nhuan District, Ho Chi Minh City, Vietnam",
    "Tan Binh District, Ho Chi Minh City, Vietnam",
    "Tan Phu District, Ho Chi Minh City, Vietnam",
    "Go Vap District, Ho Chi Minh City, Vietnam",
    "Binh Tan District, Ho Chi Minh City, Vietnam",
    "Hoc Mon District, Ho Chi Minh City, Vietnam",
    "Cu Chi District, Ho Chi Minh City, Vietnam",
    "Binh Chanh District, Ho Chi Minh City, Vietnam",
    "Nha Be District, Ho Chi Minh City, Vietnam",
    "Can Gio District, Ho Chi Minh City, Vietnam",
]

for district in districts_hcm:
    try:
        print(f"Trying to download: {district}")
        G = ox.graph_from_place(district, network_type='drive')
        print(f"✅ Successfully downloaded {district}: {len(G.nodes)} nodes, {len(G.edges)} edges\n")
    except Exception as e:
        print(f"❌ Failed to download {district}: {e}\n")
