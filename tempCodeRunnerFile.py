 request.get_json()
        product_id = data.get('product_id')
        content_w = data.get('content_w', 0.7)
        item_w = data.get('item_w', 0.3)
        k_out = data.get('k_out', 10) # total items will be display

        # Kiểm tra đầu vào hợp lệ
        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        if not (0 <= content_w <= 1 and 0 <= item_w <= 1):
            return jsonify({"error": "Weights must be between 0 and 1"}), 400

        # Gọi mô hình để dự đoán
        recommendations = Combined(product_id, content_w, item_w, k_out)
        return jsonify({"product_id": product_id, "recommendations": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


