import 'location.dart';
import 'networking.dart';

class MarkerModel {
  Future<dynamic> getLocationMarker() async {
    Location location = Location();
    await location.getCurrentLocation();

    NetworkHelper networkHelper = NetworkHelper('');
    var markerData = await networkHelper.getData();
    return markerData;
  }
}