import 'package:flutter/material.dart';


class IconContent extends StatelessWidget {

  IconContent({
    @required this.topicon,
    @required this.bottomicon,
    @required this.colour,
  });
  final IconData topicon;
  final IconData bottomicon;
  final Color colour;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          height: 10,
        ),
        Icon(
          topicon,
          size: 30,
          color: colour,
        ),
        Icon(
          bottomicon,
          size: 30,
          color: colour,
        )
      ],
    );
  }
}

