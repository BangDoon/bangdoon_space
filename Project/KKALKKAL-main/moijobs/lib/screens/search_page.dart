import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:async';
import 'dart:convert';
import 'new_post.dart';

class Note {
  String region;
  String city;
  String title;
  String link;

  Note(this.region, this.city, this.title, this.link);

  Note.fromJSON(Map<String, dynamic> json) {
    region = json['region'];
    city = json['city'];
    title = json['title'];
    link = json['link'];
  }
}

class SearchExample extends StatefulWidget {
  @override
  _SearchExampleState createState() => _SearchExampleState();
}

class _SearchExampleState extends State<SearchExample> {
  List<Note> _notes = List<Note>();
  List<Note> _notesForDisplay = List<Note>();

  Future<List<Note>> fetchNotes() async {
    var url = 'https://jsonplaceholder.typicode.com/posts';
    var response = await http.get(url);

    var notes = List<Note>();

    if (response.statusCode == 200) {
      var notesJson = jsonDecode(utf8.decode(response.bodyBytes));
      for (var noteJson in notesJson) {
        notes.add(Note.fromJSON(noteJson));
      }
    }
    return notes;
  }

  @override
  void initState() {
    fetchNotes().then((value) {
      setState(() {
        _notes.addAll(value);
        _notesForDisplay = _notes;
      });
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFF209FA6),
        title: Text('검색'),
      ),
      body: ListView.builder(
        itemBuilder: (context, index) {
          return index == 0 ? _searchBar() : _listItem(index - 1);
        },
        itemCount: _notesForDisplay.length + 1,
      ),
    );
  }

  _searchBar() {
    return Container(
      decoration: BoxDecoration(
        color: Color(0xFF209FA6),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(20),
          bottomRight: Radius.circular(20),
        ),
      ),
      padding: EdgeInsets.fromLTRB(
        15,
        10,
        15,
        10,
      ),
      child: Row(
        children: [
          Expanded(
            flex: 5,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20,vertical: 10),
              child: TextField(
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.white,
                  prefixIcon: Icon(
                    Icons.search,
                    color: Color(0xFF1A202C),
                    size: 20,
                  ),
                  hintText: '제목 검색',
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(
                      color: Colors.transparent,
                    ),
                    borderRadius: BorderRadius.all(
                      Radius.circular(20),
                    ),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderSide: BorderSide(
                      color: Colors.transparent,
                    ),
                    borderRadius: BorderRadius.all(
                      Radius.circular(20),
                    ),
                  ),
                  border: OutlineInputBorder(
                    borderSide: BorderSide(
                      color: Colors.transparent,
                    ),
                    borderRadius: BorderRadius.all(
                      Radius.circular(
                        20,
                      ),
                    ),
                  ),
                ),
                onChanged: (text) {
                  // text = text.toLowerCase();
                  setState(() {
                    _notesForDisplay = _notes.where((note) {
                      var noteTitle = note.title.toLowerCase();
                      return noteTitle.contains(text);
                    }).toList();
                  });
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  _listItem(index) {
    if (_notesForDisplay[index].city != null) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        child: Material(
          borderRadius: BorderRadius.circular(20),
          elevation: 5.0,
          child: Padding(
            padding: const EdgeInsets.all(10.0),
            child: FlatButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (BuildContext context) {
                      return NewPost(
                        _notesForDisplay[index].title,
                        _notesForDisplay[index].link,
                      );
                    },
                  ),
                );
                setState(() {});
              },
              child: Column(
                children: [
                  Row(
                    children: [
                      Text('[${_notesForDisplay[index].region}]'),
                      Text('[ ${_notesForDisplay[index].city}]'),
                    ],
                  ),
                  Text('${_notesForDisplay[index].title}'),
                ],
              ),
            ),
          ),
        ),
      );
    } else {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        child: Material(
          borderRadius: BorderRadius.circular(20),
          elevation: 5.0,
          child: Padding(
            padding: const EdgeInsets.all(10.0),
            child: FlatButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (BuildContext context) {
                      return NewPost(
                        _notesForDisplay[index].title,
                        _notesForDisplay[index].link,
                      );
                    },
                  ),
                );
                setState(() {});
              },
              child: Column(
                children: [
                  Row(
                    children: [
                      Text('[${_notesForDisplay[index].region}]'),
                    ],
                  ),
                  Text('${_notesForDisplay[index].title}'),
                ],
              ),
            ),
          ),
        ),
      );
    }
  }
}
