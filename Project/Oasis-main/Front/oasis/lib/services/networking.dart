import 'package:http/http.dart' as http;
import 'dart:convert';

class NetworkHelper {
  NetworkHelper(this.url);

  final String url;

  Future<List> getData() async {
    http.Response response = await http.get(url);

    if (response.statusCode == 200) {
      List data = jsonDecode(utf8.decode(response.bodyBytes));
      return data;
    } else {
      print(response.statusCode);
    }
  }
}
